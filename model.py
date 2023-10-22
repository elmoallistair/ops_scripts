from sklearn.model_selection import train_test_split
import os
import joblib
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def split_dataset(df, feature, target, test_size=0.2, random_state=None):
    """
    Split a DataFrame into training and testing sets with stratification.

    Parameters:
    - df: The input DataFrame.
    - feature: The name of the column(s) to be used as features (X).
    - target: The name of the column to be used as the target (y).
    - test_size: The proportion of the dataset to include in the testing split (default is 0.2).
    - random_state: An optional random seed for reproducibility (default is None).

    Returns:
    - df_train: The training DatsaFrame.
    - df_test: The testing DataFrame.
    """

    X_train, X_test, *_ = train_test_split(df[feature], df[target], test_size=test_size, 
                                                        stratify=df[target], random_state=random_state)

    df_train = df.loc[X_train.index]
    df_test = df.loc[X_test.index]

    return df_train, df_test

def train_model(X, y, model, vectorizer):
    """
    Train a machine learning model using text data and a vectorizer.

    Parameters:
    - X: A list or array of text data to be transformed and used as features.
    - y: A list or array of target labels.
    - model: The machine learning model to be trained.
    - vectorizer: The feature vectorizer for transforming text data into numerical features.

    Returns:
    - model: The trained machine learning model.
    - vectorizer: The feature vectorizer.
    """
    X_transformed = vectorizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, 
                                                        random_state=42, stratify=y)

    model.fit(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"[INFO] Model Score: {test_score:.2F}")

    return model, vectorizer

def evaluate_model(X, y_true, model, feature):
    """
    Evaluate a machine learning model's performance using text data and a feature vectorizer.

    Parameters:
    - X: Dataset a to be transformed and used for predictions.
    - y_true: A list or array of the true target labels for the provided text data.
    - model: The trained machine learning model to be evaluated.
    - feature: The feature vectorizer used to transform the text data.

    Returns:
    - df_result: A DataFrame containing review text, true labels, and model predictions.
    - df_matrices: A confusion matrix for model performance assessment.
    - df_performance: A classification report with performance metrics for each class.

    Example usage:
    # Assuming you have a list of text data 'X', true labels 'y_true', and a trained model and feature
    df_result, df_matrices, df_performance = evaluate_model(X, y_true, trained_model, features)
    """
    X_transformed = feature.transform(X)
    y_pred = model.predict(X_transformed)

    accuracy = accuracy_score(y_true, y_pred)
    df_result = pd.DataFrame({'review': X, 'tag_transform': y_true, 'prediction': y_pred})
    df_matrices = confusion_matrix(y_true, y_pred)
    df_performance = classification_report(y_true, y_pred, target_names=model.classes_)
    print(f'[INFO] Model successfully evaluated with score : {accuracy:.2f}')

    return df_result, df_matrices, df_performance

def save_model_and_feature(model, feature, base_path, task_id, model_id, backup=True):
    """
    Save a machine learning model, its associated feature vectorizer, and identifiers to specified directories.

    Parameters:
    - model: The trained machine learning model to be saved.
    - feature: The feature vectorizer used for text data transformation.
    - base_path: The base directory path where the 'latest' and 'backup' directories will be created.
    - task_id: An identifier for the task or purpose.
    - model_id: An identifier for the algorithm or model name.
    - backup: If True, create backup copies of the model, feature, and vectorizer files.
    """

    latest_path = os.path.join(base_path, 'latest')
    os.makedirs(latest_path, exist_ok=True)

    model_latest_path = os.path.join(latest_path, f'{task_id}_model_latest_{model_id}.joblib')
    feature_latest_path = os.path.join(latest_path, f'{task_id}_feature_latest_{model_id}.joblib')

    if os.path.exists(model_latest_path) or os.path.exists(feature_latest_path):
        print(f"[WARNING] The model or feature already exist in 'latest' and is being overwritten..")


    joblib.dump(model, model_latest_path)
    joblib.dump(feature, feature_latest_path)

    print(f'[SUCCESS] Model and feature saved to {base_path}/latest')

    if backup:
        current_date = datetime.now().strftime('%Y-%m-%d')
        backup_path = os.path.join(base_path, 'backup')

        os.makedirs(backup_path, exist_ok=True)
        model_backup_path = os.path.join(backup_path, f'{task_id}_model_{current_date}.joblib')
        feature_backup_path = os.path.join(backup_path, f'{task_id}_feature_{current_date}.joblib')

        if os.path.exists(model_backup_path) or os.path.exists(feature_backup_path):
            print(f"[WARNING] The model or feature already exist in 'backup' and is being overwritten..")

        joblib.dump(model, model_backup_path)
        joblib.dump(feature, feature_backup_path)

        print(f'[SUCCESS] Backup of model and feature saved to {base_path}/backup')
