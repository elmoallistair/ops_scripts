import os
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def split_dataframe_stratified(df, feature, target, test_size=0.2, random_state=None):
    """
    Split a DataFrame into training and testing sets with stratification.

    Parameters:
    - df: The input DataFrame.
    - feature: The name of the column(s) to be used as features (X).
    - target: The name of the column to be used as the target (y).
    - test_size: The proportion of the dataset to include in the testing split (default is 0.2).
    - random_state: An optional random seed for reproducibility (default is None).

    Returns:
    - df_train: The training DataFrame.
    - df_test: The testing DataFrame.
    """

    X_train, X_test, *_ = train_test_split(df[feature], df[target], test_size=test_size, 
                                                        stratify=df[target], random_state=random_state)

    df_train = df.loc[X_train.index]
    df_test = df.loc[X_test.index]

    return df_train, df_test

def vectorize_dataset(df, feature):
    """
    Train a TF-IDF vectorizer on a DataFrame and apply it to the same DataFrame.

    Parameters:
    - df: The input DataFrame containing both training and testing data.
    - feature: The name of the column containing text data (X).

    Returns:
    - df_tfidf: TF-IDF vectors for the text data in the input DataFrame.
    """

    tfidf_vectorizer = TfidfVectorizer()
    df_tfidf = tfidf_vectorizer.fit_transform(df[feature])

    return df_tfidf

def train_model(X_train_tfidf, y_train, model):
    """
    Train a classification model on TF-IDF vectors.

    Parameters:
    - X_train_tfidf: TF-IDF vectors for the training text data.
    - y_train: The target variable for the training data.
    - model: The classification model to be trained.

    Returns:
    - trained_model: The trained classification model.
    """

    trained_model = model.fit(X_train_tfidf, y_train)

    return trained_model

def save_model_and_feature(model, feature, base_path, identifier, backup=True):
    """
    Save a machine learning model and its associated feature vectorizer to specified directories.
    
    Parameters:
    - model: The trained machine learning model to be saved.
    - feature: The feature vectorizer used for text data transformation.
    - base_path: The base directory path where the 'latest' and 'backup' directories will be created.
    - identifier: A unique identifier for the model and feature files.
    - backup: If True, create backup copies of the model and feature files.
    
    The function saves the model and feature in the 'latest' directory and, if specified, in the 'backup' directory. 
    It checks for file existence and provides warnings if files are being overwritten.
    
    Example usage:
    base_path = 'models'
    save_model_and_feature(trained_model, features, base_path, identifier='deli_prt', backup=True)
    """
    def check_file_exist(file_path, directory_name):
        filename = os.path.basename(file_path)
        if os.path.isfile(file_path):
            print(f"[WARNING] The '{filename}' file in the '{directory_name}' directory already exists and is being overwritten.")

    latest_path = os.path.join(base_path, 'latest')
    os.makedirs(latest_path, exist_ok=True)

    model_latest_path = os.path.join(latest_path, f'{identifier}_model_latest.joblib')
    feature_latest_path = os.path.join(latest_path, f'{identifier}_feature_latest.joblib')

    joblib.dump(model, model_latest_path)
    joblib.dump(feature, feature_latest_path)

    check_file_exist(model_latest_path, 'latest')
    check_file_exist(feature_latest_path, 'latest')
    print(f'[SUCCESS] Model and feature saved to {base_path}/latest')

    if backup:
        current_date = datetime.now().strftime('%Y-%m-%d')
        backup_path = os.path.join(base_path, 'backup')

        os.makedirs(backup_path, exist_ok=True)
        model_backup_path = os.path.join(backup_path, f'{identifier}_model_{current_date}.joblib')
        feature_backup_path = os.path.join(backup_path, f'{identifier}_feature_{current_date}.joblib')

        joblib.dump(model, model_backup_path)
        joblib.dump(feature, feature_backup_path)

        check_file_exist(model_backup_path, 'backup')
        check_file_exist(feature_backup_path, 'backup')
    
        print(f'[SUCCESS] Backup of model and feature saved to {base_path}/backup')
