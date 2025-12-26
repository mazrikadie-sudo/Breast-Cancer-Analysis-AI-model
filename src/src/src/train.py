from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
    return model

def save_model(model, scaler):
    joblib.dump(model, "models/trained_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
