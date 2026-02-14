import serial
import time
import json
import joblib
import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
SERIAL_PORT = 'COM3'   # CHANGE THIS to your Arduino Port! (e.g., '/dev/ttyUSB0')
BAUD_RATE = 115200
JSON_FILE = "piezo.json"
MODEL_FILE = "stampede_model.pkl"

# Load AI Model
print("Loading AI Model...")
if os.path.exists(MODEL_FILE):
    clf = joblib.load(MODEL_FILE)
else:
    print(f"❌ Error: {MODEL_FILE} not found. Train your model first!")
    exit()

# Connect to Arduino
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"✅ Connected to {SERIAL_PORT}")
except:
    print(f"❌ Could not connect to {SERIAL_PORT}")
    exit()

data_buffer = []
WINDOW_SIZE = 50 # 0.5 seconds of data (Faster response for demo)

print("⚡ Piezo Node Started. Tap the sensor...")

while True:
    if ser.in_waiting > 0:
        try:
            line = ser.readline().decode('utf-8').strip()
            parts = line.split(',')
            
            if len(parts) == 2:
                val = int(parts[1])
                data_buffer.append(val)
                
                # If we have enough data, PREDICT
                if len(data_buffer) >= WINDOW_SIZE:
                    # Feature Extraction (Must match training!)
                    df = pd.DataFrame({'Voltage': data_buffer})
                    features = [[
                        df['Voltage'].max(),
                        df['Voltage'].std(),
                        df['Voltage'].min(),
                        (df['Voltage'] > 50).sum() # Threshold impacts
                    ]]
                    
                    # Get Probability of Class 1 (Stampede)
                    risk_score = clf.predict_proba(features)[0][1]
                    
                    # Write to File
                    try:
                        with open(JSON_FILE, "w") as f:
                            json.dump({"risk": risk_score}, f)
                            f.flush()
                            os.fsync(f.fileno())
                    except:
                        pass

                    # Print status
                    bar = "█" * int(risk_score * 10)
                    print(f"Piezo Risk: {risk_score:.2f} | {bar}")

                    # Reset Buffer (Sliding window overlap could be better, but this is simple)
                    data_buffer = [] 

        except Exception as e:
            pass # Ignore glitches