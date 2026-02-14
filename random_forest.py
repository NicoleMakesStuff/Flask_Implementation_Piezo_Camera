import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. LOAD DATA
# Make sure this matches the filename you generated
df = pd.read_csv("data.csv")

# 2. FEATURE EXTRACTION (The "Smart" Part)
# We chop the continuous stream into 1-second "Windows"
WINDOW_SIZE = 100 # 100 samples = 1 second (since sampling rate was 100Hz)

X = [] # Features
y = [] # Labels (0 or 1)

# Loop through the data in chunks
for i in range(0, len(df) - WINDOW_SIZE, WINDOW_SIZE):
    window = df.iloc[i : i + WINDOW_SIZE]
    
    # Calculate Statistical Features for this 1-second chunk
    # These "summarize" the physics of that second.
    features = [
        window['Voltage'].max(),       # Peak Force (Stomp?)
        window['Voltage'].std(),       # Chaos/Energy Level
        window['Voltage'].min(),       # Negative rebound
        (window['Voltage'] > 0.5).sum() # Count of strong impacts (Frequency)
    ]
    
    # The label is the most common label in this window
    label = window['Label'].mode()[0]
    
    X.append(features)
    y.append(label)

X = np.array(X)
y = np.array(y)

print(f"✅ Created {len(X)} training samples (1-second windows).")

# 3. SPLIT TRAIN / TEST
# 80% for training, 20% to prove to judges it works
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. TRAIN RANDOM FOREST
print("Training Random Forest Model...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 5. EVALUATE
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n🎯 Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Safe", "Stampede"]))

# 6. SAVE THE MODEL
# You can load this file ('stampede_model.pkl') in your live demo script!
joblib.dump(clf, "stampede_model.pkl")
print("💾 Model saved as 'stampede_model.pkl'")

# --- SHOW FEATURE IMPORTANCE ---
# This tells you WHAT the AI is looking at
importances = clf.feature_importances_
feature_names = ["Max Voltage", "Std Dev", "Min Voltage", "Impact Count"]
for name, importance in zip(feature_names, importances):
    print(f"Feature '{name}': {importance:.4f}")