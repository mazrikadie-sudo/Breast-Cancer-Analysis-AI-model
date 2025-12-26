import joblib
import numpy as np

def predict(sample):
    model = joblib.load("models/trained_model.pkl")
    scaler = joblib.load("models/scaler.pkl")

    sample = np.array(sample).reshape(1, -1)
    sample = scaler.transform(sample)

    prediction = model.predict(sample)
    return "Malignant" if prediction[0] == 0 else "Benign"
