import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('mission_model.h5')

# Simulated embedded inference scenario
sample_input = np.array([[0.5, 0.7, 0.8, 0.3, 0.6]])  # random example
prediction = model.predict(sample_input)[0][0]

print(f"Threat prediction probability: {prediction:.2f}")
print("Threat detected!" if prediction > 0.5 else "No immediate threat.")
