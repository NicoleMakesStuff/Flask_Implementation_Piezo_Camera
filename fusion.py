import json
import time
import os

print("--- 🧠 DYNAMIC FUSION ENGINE 🧠 ---")

def read_json(filename):
    if not os.path.exists(filename): return {}
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except:
        return {}

while True:
    # 1. READ DATA
    cam_data = read_json("camera.json")
    piezo_data = read_json("piezo.json")

    c_risk = cam_data.get("risk", 0.0)
    c_conf = cam_data.get("confidence", 1.0) # Default to 1.0 if missing
    
    p_risk = piezo_data.get("risk", 0.0)
    
    # 2. DYNAMIC WEIGHT CALCULATION
    # Base assumption: Camera is better (0.6)
    # But we scale it by confidence.
    
    w_cam = 0.6 * c_conf
    
    # If camera is blind (c_conf=0.1), w_cam becomes 0.06
    # So we give the remaining weight to Piezo
    w_piezo = 1.0 - w_cam
    
    # 3. CALCULATE FUSED RISK
    total_risk = (c_risk * w_cam) + (p_risk * w_piezo)
    
    # 4. DISPLAY
    # Visual bar for weights to show judges "The Brain is Thinking"
    weight_bar = f"Cam: {int(w_cam*100)}% | Piezo: {int(w_piezo*100)}%"
    
    if c_conf < 0.5:
        system_status = "⚠️ CAM BLIND - RELYING ON SENSORS"
        color = "\033[93m" # Yellow
    else:
        system_status = "✅ SYSTEM OPTIMAL"
        color = "\033[92m" # Green

    if total_risk > 0.75:
        alert = "🚨 CRITICAL STAMPEDE"
        alert_color = "\033[91m" # Red
    else:
        alert = "Safe"
        alert_color = "\033[97m" # White

    print(f"{color}[STATUS] {system_status}\033[0m")
    print(f"   weights: [{weight_bar}]")
    print(f"   RISK:    {alert_color}{total_risk:.2f} ({alert})\033[0m")
    print("-" * 40)
    
    time.sleep(0.5)