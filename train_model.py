# train_model.py
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib

def train_classifier(data_path: str, model_path: str):
    """
    Trains a LightGBM classifier on the extracted features and saves the model.
    """
    df = pd.read_csv(data_path)
    X = df.drop(['text', 'label'], axis=1)
    y = df['label']

    # --- MODIFIED CODE BLOCK ---
    # Convert boolean columns to integers (0 or 1) instead of categories.
    # This is a more robust way to handle boolean features.
    for col in X.select_dtypes(include=['bool']).columns:
        X[col] = X[col].astype(int)
    # --- END OF MODIFIED CODE BLOCK ---

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print("Training LightGBM model...")
    # The 'categorical_feature' parameter is no longer needed
    model = lgb.LGBMClassifier(objective='multiclass', random_state=42)
    model.fit(X_train, y_train)

    print("\n--- Model Evaluation ---")
    y_pred = model.predict(X_test)
    all_labels = range(len(label_encoder.classes_))

    print(classification_report(
        y_test, y_pred, labels=all_labels,
        target_names=label_encoder.classes_, zero_division=0
    ))

    print(f"\nSaving model to '{model_path}'...")
    joblib.dump({'model': model, 'label_encoder': label_encoder}, model_path)
    print("✅ Model training complete.")

if __name__ == '__main__':
    training_data_file = 'training_data.csv'
    output_model_file = 'document_structure_model.joblib'
    train_classifier(training_data_file, output_model_file)