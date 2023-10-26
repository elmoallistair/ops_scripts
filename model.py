from sklearn.model_selection import train_test_split
import os
import joblib
import pandas as pd
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

def evaluate_model(X, y_true, model, feature, class_weights=None):
    """
    Evaluate a machine learning model's performance using text data and a feature vectorizer.

    Parameters:
    - X: Dataset to be transformed and used for predictions.
    - y_true: A list or array of the true target labels for the provided text data.
    - model: The trained machine learning model to be evaluated.
    - feature: The feature vectorizer used to transform the text data.
    - class_weights: A dictionary of class weights if you want to calculate weighted average.

    Returns:
    - df_result: A DataFrame containing review text, true labels, model predictions, and confidence scores.
    - df_matrices: A confusion matrix for model performance assessment.
    - df_performance: A DataFrame containing performance metrics for each class, accuracy, macro avg, and weighted avg.
    """
    X_transformed = feature.transform(X)
    y_pred = model.predict(X_transformed)
    
    if hasattr(model, 'predict_proba'):
        confidence_scores = model.predict_proba(X_transformed).max(axis=1)
    else:
        confidence_scores = [None] * len(y_true)

    accuracy = accuracy_score(y_true, y_pred)
    df_result = pd.DataFrame({'review': X, 'actual': y_true, 'prediction': y_pred, 'confidence': confidence_scores})
    df_result['confidence'] = df_result['confidence'].apply(lambda x: f'{x:.4f}' if x is not None else None)
    df_matrices = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_weights.keys() if class_weights else None, output_dict=True)
    
    if class_weights:
        weighted_avg = {}
        for class_name, weights in class_weights.items():
            precision = report[class_name]['precision']
            recall = report[class_name]['recall']
            f1 = report[class_name]['f1-score']
            support = report[class_name]['support']
            
            weighted_precision = precision * support
            weighted_recall = recall * support
            weighted_f1 = f1 * support
            
            weighted_avg[class_name] = {
                'precision': f'{weighted_precision:.4f}',
                'recall': f'{weighted_recall:.4f}',
                'f1-score': f'{weighted_f1:.4f}',
                'support': str(support)
            }

        total_weight = sum([sum(weights.values()) for weights in class_weights.values()])
        for metric in ['precision', 'recall', 'f1-score']:
            weighted_avg['weighted avg'][metric] = f'{(sum([weights[metric] for weights in class_weights.values()]) / total_weight):.4f}'
        weighted_avg['weighted avg']['support'] = str(sum([weights['support'] for weights in class_weights.values()]))
        report['weighted avg'] = weighted_avg
    
    df_performance = pd.DataFrame.from_dict(report).transpose()
    print(f'[INFO] Model successfully evaluated with score: {accuracy:.4f}')
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
