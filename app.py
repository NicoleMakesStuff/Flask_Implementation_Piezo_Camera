import cv2
import time
import threading
import serial
import joblib
import pandas as pd
import numpy as np
import os
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO

# --- CONFIGURATION ---
SERIAL_PORT = 'COM3'   # CHECK YOUR ARDUINO PORT
BAUD_RATE = 115200
SNAPSHOT_INTERVAL = 1.0  # Take a photo every 1 second
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "stampede_model.pkl")
app = Flask(__name__)

# --- GLOBAL SHARED STATE ---
# This dictionary replaces your multiple JSON files
system_state = {
    "camera": {
        "person_count": 0,
        "risk": 0.0,
        "confidence": 1.0,  # 1.0 = Clear, 0.1 = Blocked
        "status": "CLEAR",
        "last_update": 0
    },
    "piezo": {
        "risk": 0.0,
        "raw_val": 0
    },
    "fusion": {
        "total_risk": 0.0,
        "alert": "SAFE",
        "weights": {"cam": 0.6, "piezo": 0.4}
    }
}

# Buffer to store the latest processed snapshot
last_processed_frame = None
frame_lock = threading.Lock()

# --- 1. PIEZO WORKER (Background Thread) ---
def piezo_worker():
    print("⚡ Piezo Thread Started...")
    
    # Load AI Model
    clf = None
    try:
        clf = joblib.load(MODEL_FILE)
        print("✅ Piezo AI Model Loaded")
    except Exception as e:
        print("❌ Model loading failed!")
        print("Path tried:", MODEL_FILE)
        print("Error:", e)


    # Connect Serial
    ser = None
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"✅ Arduino Connected on {SERIAL_PORT}")
    except:
        print("❌ Arduino not found. Using Mock Data mode.")

    data_buffer = []
    WINDOW_SIZE = 50 

    while True:
        try:
            val = 0
            # Read from Serial or Mock
            if ser and ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                parts = line.split(',')
                if len(parts) == 2:
                    val = int(parts[1])
            elif ser is None:
                time.sleep(0.05)
                val = np.random.randint(0, 10) # Mock noise

            # Buffer Logic
            data_buffer.append(val)
            if len(data_buffer) > WINDOW_SIZE:
                data_buffer.pop(0)

            # Prediction
            risk = 0.0
            if len(data_buffer) == WINDOW_SIZE and clf:
                df = pd.DataFrame({'Voltage': data_buffer})
                features = [[
                    df['Voltage'].max(),
                    df['Voltage'].std(),
                    df['Voltage'].min(),
                    (df['Voltage'] > 50).sum()
                ]]
                risk = clf.predict_proba(features)[0][1]
            else:
                # Fallback if no model
                risk = min(1.0, max(data_buffer) / 100.0)

            # Update State
            system_state["piezo"]["risk"] = float(risk)
            system_state["piezo"]["raw_val"] = val
            
        except Exception as e:
            pass # Ignore serial glitches

# --- 2. CAMERA SNAPSHOT WORKER (Background Thread) ---
def camera_snapshot_worker():
    global last_processed_frame
    print("📷 Camera Snapshot Thread Started...")
    
    cap = cv2.VideoCapture(0) # Open Camera
    model = YOLO('yolov8n.pt')

    while True:
        # 1. Capture Snapshot
        ret, frame = cap.read()
        if not ret:
            time.sleep(3)
            continue

        # 2. Visibility Check (Blind/Dark Detection)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        confidence = 1.0
        if brightness < 30 or contrast < 10:
            confidence = 0.1 # Camera is blocked or dark
            cv2.putText(frame, "⚠️ POOR VISIBILITY", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        
        # 3. YOLO Detection
        count = 0
        if confidence > 0.5:
            results = model(frame, verbose=False)
            for result in results:
                for box in result.boxes:
                    if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.4:
                        count += 1
                        x1,y1,x2,y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        
        # 4. Update Global State
        system_state["camera"]["person_count"] = count
        system_state["camera"]["confidence"] = confidence
        system_state["camera"]["risk"] = min(1.0, count / 8.0) # 8 people = 100% risk
        system_state["camera"]["last_update"] = time.time()

        # 5. Encode Image for Frontend
        ret, buffer = cv2.imencode('.jpg', frame)
        with frame_lock:
            last_processed_frame = buffer.tobytes()

        # 6. Sleep (Simulating Snapshot Interval)
        time.sleep(SNAPSHOT_INTERVAL)

# --- 3. FLASK ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/latest_image')
def latest_image():
    """Returns the latest processed snapshot as a static image"""
    with frame_lock:
        if last_processed_frame:
            return Response(last_processed_frame, mimetype='image/jpeg')
        else:
            return "Camera Initializing...", 503

@app.route('/api/fusion_stats')
def get_fusion_stats():
    """Merges data and returns result to frontend"""
    
    # Get Data
    c_risk = system_state["camera"]["risk"]
    c_conf = system_state["camera"]["confidence"]
    p_risk = system_state["piezo"]["risk"]
    
    # --- FUSION LOGIC ---
    # Weight Adjustment: If camera is blind, trust Piezo more.
    w_cam = 0.6 * c_conf
    w_piezo = 1.0 - w_cam
    
    total_risk = (c_risk * w_cam) + (p_risk * w_piezo)
    
    status = "SAFE"
    if total_risk > 0.8: status = "CRITICAL"
    elif total_risk > 0.5: status = "WARNING"
    
    response_data = {
        "camera": system_state["camera"],
        "piezo": system_state["piezo"],
        "fusion": {
            "total_risk": round(total_risk, 2),
            "status": status,
            "weights": {
                "cam": round(w_cam, 2),
                "piezo": round(w_piezo, 2)
            }
        }
    }
    return jsonify(response_data)

if __name__ == '__main__':
    # Start Background Threads
    threading.Thread(target=piezo_worker, daemon=True).start()
    threading.Thread(target=camera_snapshot_worker, daemon=True).start()
    
    # Run Web Server
    app.run(host='0.0.0.0', port=5001, debug=False)