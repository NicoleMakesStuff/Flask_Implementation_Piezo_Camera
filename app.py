import cv2
import time
import threading
import serial
import json
import joblib
import pandas as pd
import numpy as np
import os
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO

# --- CONFIGURATION ---
SERIAL_PORT = 'COM7'   # CHECK YOUR ARDUINO PORT
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
    
    # Load AI Model (Not used for heuristic, but kept)
    clf = None
    try:
        if os.path.exists(MODEL_FILE):
            clf = joblib.load(MODEL_FILE)
            print("✅ Piezo AI Model Loaded")
    except Exception as e:
        print("❌ Model loading failed:", e)

    # Initialize Serial
    ser = None
    
    while True:
        # Reconnect Logic
        if ser is None:
            try:
                ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
                print(f"✅ Arduino Connected on {SERIAL_PORT}")
            except:
                print(f"❌ Waiting for Arduino on {SERIAL_PORT}...")
                time.sleep(2)
                continue

        try:
            line = ""
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
            
            # Parsing Logic
            if not line: continue
            
            # --- PARSING ---
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                # print("JSON Parse Error:", line)
                continue

            # Extract Values from Parts
            dist = float(data.get('D', 100.0))
            vib = int(data.get('V', 0))
            ir = int(data.get('IR', 0))
            pir = int(data.get('PIR', 0))
            piezos = [float(data.get(f'P{i}', 0.0)) for i in range(1, 6)]

            # --- RISK CALCULATION ---
            score_dist = 1.0 if dist < 50 else 0.0
            score_vib = 1.0 if vib == 1 else 0.0
            score_motion = 1.0 if (ir == 1 or pir == 1) else 0.0
            total_pressure = sum(piezos)
            score_piezo = min(1.0, total_pressure / 50.0)

            sensor_risk = (0.2 * score_dist) + \
                          (0.2 * score_vib) + \
                          (0.2 * score_motion) + \
                          (0.4 * score_piezo)
            
            # Update State
            system_state["piezo"]["risk"] = float(sensor_risk)
            system_state["piezo"]["raw_data"] = data
            
            # Debug Print
            # print(f"Risk: {sensor_risk:.2f}")

        except Exception as e:
            print(f"❌ Serial Error: {e}")
            if ser:
                try:
                    ser.close()
                except:
                    pass
            ser = None # Trigger Reconnect
            time.sleep(1) 

# --- 2. CAMERA SNAPSHOT WORKER (Background Thread) ---
def camera_snapshot_worker():
    global last_processed_frame
    print("📷 Camera Snapshot Thread Started...")
    
    # Try Index 0 with DirectShow (Better for Windows)
    print("Trying Camera Index 0...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened() or not cap.read()[0]:
        print("⚠️ Index 0 failed/empty. Trying Index 1...")
        cap.release()
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
    if not cap.isOpened():
        print("❌ CRITICAL ERROR: Could not open ANY camera.")
    else:
        print(f"✅ Camera Opened Successfully")

    model = YOLO('yolov8n.pt')
    print("✅ YOLO Model Loaded")

    while True:
        # 1. Capture Snapshot
        # print("Attempting read...") 
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Warning: Failed to read frame from camera. Retrying...")
            time.sleep(3)
            continue
        
        # print(f"✅ Frame Captured: {frame.shape}") # Verify we actually get data
        
        # 2. Visibility Check (Blind/Dark/Blur Detection)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # Blur Detection using Laplacian Variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Debug Info on Frame
        cv2.putText(frame, f"B:{brightness:.1f} C:{contrast:.1f} Blur:{laplacian_var:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        confidence = 1.0
        status_text = ""
        
        # Check for poor visibility conditions
        if brightness < 10 or contrast < 5:
            confidence = 0.1
            status_text = "POOR VISIBILITY - DARK"
        elif laplacian_var < 100:  # Threshold for blur detection (adjust as needed)
            confidence = 0.1
            status_text = "POOR VISIBILITY - BLURRY"
        
        if status_text:
            cv2.putText(frame, status_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            system_state["camera"]["status"] = "BLOCKED"
        else:
            system_state["camera"]["status"] = "CLEAR"
        
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
    if c_conf > 0.5:
        w_cam = 0.6
        w_sensor = 0.4
    else:
        w_cam = 0.1
        w_sensor = 0.9
    
    total_risk = (c_risk * w_cam) + (p_risk * w_sensor)
    
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
                "piezo": round(w_sensor, 2)
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