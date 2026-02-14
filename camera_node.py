import cv2
import json
import numpy as np
import os
from ultralytics import YOLO

# --- CONFIGURATION ---
JSON_FILE = "camera.json"
model = YOLO('yolov8n.pt') 
cap = cv2.VideoCapture(0)

def calculate_visibility(frame):
    """
    Returns a score 0.0 (Blind/Dark/Fog) to 1.0 (Clear).
    """
    # Convert to Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 1. Calculate Brightness (Mean Intensity)
    brightness = np.mean(gray)
    
    # 2. Calculate Contrast (Standard Deviation)
    # Low contrast usually means fog or a covered lens
    contrast = np.std(gray)
    
    # Normalize:
    # If brightness < 30 (Dark), score drops.
    # If contrast < 10 (Flat Gray/Fog), score drops.
    
    if brightness < 30 or contrast < 10:
        return 0.1 # Low Confidence (Blind)
    else:
        return 1.0 # High Confidence (Clear)

print("📷 Smart Camera Node Started...")

while True:
    ret, frame = cap.read()
    if not ret: break

    # --- 1. CHECK VISIBILITY ---
    visibility_score = calculate_visibility(frame)
    
    # --- 2. RUN YOLO (Only if visible) ---
    person_count = 0
    if visibility_score > 0.5:
        results = model(frame, stream=True, verbose=False)
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.5:
                    person_count += 1
                    # Draw box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    else:
        # Draw "BLIND" warning on screen
        cv2.putText(frame, "⚠️ CAMERA BLOCKED / FOG", (50, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # --- 3. CALCULATE RISK ---
    # If camera is blind, Risk is 0.0 (from camera side), 
    # BUT we send low confidence so Fusion Hub knows to ignore it.
    risk_score = min(1.0, person_count / 5.0)

    # --- 4. WRITE DATA + CONFIDENCE ---
    data = {
        "risk": risk_score,
        "confidence": visibility_score  # <--- NEW FIELD
    }
    
    try:
        with open(JSON_FILE, "w") as f:
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
    except:
        pass

    # Display
    status = f"P: {person_count} | Vis: {visibility_score:.1f}"
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow('Smart Camera', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()