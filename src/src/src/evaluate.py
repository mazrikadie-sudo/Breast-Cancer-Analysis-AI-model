from sklearn.metrics import classification_report, confusion_matrix

def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    print("Classification Report:\n")
    print(classification_report(y_test, predictions))
    print("Confusion Matrix:\n")
    print(confusion_matrix(y_test, predictions))
