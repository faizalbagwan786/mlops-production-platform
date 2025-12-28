from sklearn.metrics import accuracy_score


def evaluate_model(model, X_test, y_test, threshold=0.7):
    """
    Evaluate model performance and decide whether it passes the quality gate.
    """

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Evaluation accuracy: {accuracy}")

    if accuracy >= threshold:
        print("Model passed evaluation gate")
        return True, accuracy
    else:
        print("Model failed evaluation gate")
        return False, accuracy
