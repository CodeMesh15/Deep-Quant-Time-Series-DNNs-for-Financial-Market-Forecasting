
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

def train_logistic_regression(X_train, y_train):
    """
    Initializes and trains a Logistic Regression model.

    Args:
        X_train (pd.DataFrame or np.ndarray): Training features.
        y_train (pd.Series or np.ndarray): Training target variable.

    Returns:
        LogisticRegression: The trained scikit-learn model object.
    """
    print("Training Logistic Regression baseline model...")
    
    # Initialize the model
    # We use class_weight='balanced' to handle potential imbalances in up/down days
    model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    
    # Train the model
    model.fit(X_train, y_train)
    
    print("Baseline model training complete.")
    return model

def evaluate_baseline(model, X_test, y_test):
    """
    Evaluates the trained baseline model.
    """
    print("\n--- Baseline Model Evaluation ---")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, predictions))
