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

    try:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            # Expected: D: 12.3 | V: 1 | IR: 1 | PIR: 1 | P1: 0.1 | P2: 0.2 | P3: 0.1 | P4: 0.5 | P5: 0.0
            
            try:
                # Parse Data
                data = json.loads(line)

                # Extract Values
                dist = float(data.get('D', 100.0))
                vib = int(data.get('V', 0))
                ir = int(data.get('IR', 0))
                pir = int(data.get('PIR', 0))
                piezos = [float(data.get(f'P{i}', 0.0)) for i in range(1, 6)]
                
                # --- RISK CALCULATION ---
                # 1. Distance Risk (20%): High if < 50cm
                score_dist = 1.0 if dist < 50 else 0.0
                
                # 2. Vibration Risk (20%): High if detected
                score_vib = 1.0 if vib == 1 else 0.0
                
                # 3. Motion Risk (20%): High if any motion
                score_motion = 1.0 if (ir == 1 or pir == 1) else 0.0
                
                # 4. Piezo Risk (40%): Accumulate pressure
                total_pressure = sum(piezos)
                score_piezo = min(1.0, total_pressure / 50.0)

                # Total Weighted Risk
                final_risk = (0.2 * score_dist) + \
                             (0.2 * score_vib) + \
                             (0.2 * score_motion) + \
                             (0.4 * score_piezo)

                # Write to File
                try:
                    with open(JSON_FILE, "w") as f:
                        json.dump({
                            "risk": final_risk,
                            "sensors": data
                        }, f)
                        f.flush()
                        os.fsync(f.fileno())
                except:
                    pass

                # Print status
                bar = "█" * int(final_risk * 10)
                print(f"Risk: {final_risk:.2f} | D:{dist} V:{vib} M:{ir|pir} P:{total_pressure:.1f} | {bar}")

            except ValueError:
                pass # Parsing error (incomplete line)

    except Exception as e:
        pass # Serial glitches