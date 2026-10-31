# Medical-AI-Neural-Network
An Artificial Neural Network (ANN) built using TensorFlow/Keras to predict diabetes risk using 6 clinical health metrics.
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Expand to a complex Patient Database (6 Medical Features)
# Features: [Glucose, Blood Pressure, BMI, Insulin Level, Age, Family History (0 or 1)]
raw_medical_data = {
    'Glucose': [85, 168, 110, 189, 89, 166, 95, 197, 125, 172, 99, 160],
    'BloodPressure': [66, 74, 92, 60, 80, 72, 85, 70, 96, 68, 64, 82],
    'BMI': [22.5, 33.6, 28.1, 38.0, 24.2, 25.8, 31.2, 45.8, 29.3, 35.1, 21.8, 30.4],
    'Insulin': [70, 0, 125, 200, 15, 175, 0, 543, 110, 0, 88, 140],
    'Age': [21, 54, 30, 42, 33, 51, 26, 53, 41, 43, 22, 39],
    'FamilyHistory': [0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0],
    'Outcome': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # 0 = Normal, 1 = Diabetic Risk
}

df = pd.DataFrame(raw_medical_data)

# 2. Extract Features and Target
X = df[['Glucose', 'BloodPressure', 'BMI', 'Insulin', 'Age', 'FamilyHistory']]
y = df['Outcome']

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 4. Neural Networks require Data Scaling (Normalizing values)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Build the Brain: Artificial Neural Network Architecture
model = Sequential([
    Dense(12, activation='relu', input_shape=(6,)), # Input Layer + Hidden Layer 1 (12 Neurons)
    Dense(6, activation='relu'),                    # Hidden Layer 2 (6 Neurons)
    Dense(1, activation='sigmoid')                  # Output Layer (Binary classification output 0 to 1)
])

# 6. Compile the Deep Learning Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 7. Train the Neural Network
print("--- Training the Artificial Neural Network ---")
model.fit(X_train_scaled, y_train, epochs=50, batch_size=2, verbose=1)

# 8. Evaluate performance on unseen medical data
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nNeural Network Diagnostic Accuracy: {accuracy * 100:.2f}%")

# 9. Real-Time Clinical Scanner (Test with a custom complex patient profile)
# Patient: Glucose=165, BP=80, BMI=34.2, Insulin=150, Age=46, Family History=1
custom_patient = scaler.transform([[165, 80, 34.2, 150, 46, 1]])
prediction_probability = model.predict(custom_patient)[0][0]

print("\n--- Deep Learning Risk Analysis ---")
print(f"Calculated Neural Diagnostic Probability: {prediction_probability * 100:.2f}%")
if prediction_probability > 0.5:
    print("STATUS: FLAG RAISED. Predictive AI indicates structural metabolic risks.")
else:
    print("STATUS: CLEAR. Cellular and clinical metrics fall within safe baselines.")
