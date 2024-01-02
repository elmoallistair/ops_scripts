import os
import re
import string
import joblib
import numpy as np
import pandas as pd
from flashtext import KeywordProcessor

def correct_prediction_label(label, lookup_df, lookup_column='transform', result_column='origin'):
    """
    Corrects or re-mapping the prediction label using a lookup DataFrame.

    Args:
    - label (str): The label to be corrected.
    - lookup_df (pd.DataFrame): The DataFrame used for label correction.
    - lookup_column (str, optional): The column in the lookup DataFrame used for matching labels. Default is 'transform'.
    - result_column (str, optional): The column in the lookup DataFrame that contains the corrected labels. Default is 'origin'.

    Returns:
      The corrected label if a match is found in the lookup DataFrame, otherwise returns the original label.
    """
    match = lookup_df[lookup_df[lookup_column] == label]
    if not match.empty:
        return match.iloc[0][result_column]
    else:
        return label

def check_repetitive_dax(df_tickets, identifier, dax_threshold=3, id_threshold=2):
    """
    Checks for repetitive Dax/identifiers in tickets, and updates the 'check' column accordingly.
    Args:
    - df_tickets (pd.DataFrame): The DataFrame containing the tickets data.
    - identifier (str or list): The column(s) to be used as the identifier(s) for checking repetitiveness.
    - dax_threshold (int, optional): The minimum count for a driver_id to be considered repetitive. Default is 3.
    - id_threshold (int, optional): The minimum count for an identifier to be considered repetitive. Default is 2.
    Returns:
    - df_tickets (pd.DataFrame): The DataFrame with the 'check' column updated based on repetitiveness.
    """
    df_tickets['identifier'] = df_tickets[identifier].astype(str).apply(lambda x: '_'.join(x), axis=1)

    driver_counts = df_tickets['driver_id'].value_counts()
    identifier_counts = df_tickets['identifier'].value_counts()
    df_tickets.loc[df_tickets['driver_id'].map(lambda x: driver_counts[x] >= dax_threshold), 'check'] = 'Done'
    df_tickets.loc[df_tickets['identifier'].map(lambda x: identifier_counts[x] >= id_threshold), 'check'] = 'Done'

    df_tickets['check'] = df_tickets['check'].fillna('')
    df_tickets = df_tickets.drop(columns='identifier')
    
    return df_tickets

def get_prediction_with_model(df_review, model, vectorizer, col_target='review_or_remarks'):
    """
    Predict the class and confidence score for each row in a DataFrame.

    Args:
    - df_review (pd.DataFrame): The DataFrame containing the text data.
    - model: The trained machine learning model for prediction.
    - vectorizer: The text vectorizer (e.g., CountVectorizer or TF-IDFVectorizer).
    - col_feature (str): The name of the column in the DataFrame that contains the text data.

    Returns:
      A tuple of two NumPy arrays, 
      where the first array contains the predicted tags and the second array contains the confidence scores.
    """

    X = vectorizer.transform(df_review[col_target])
    predictions = model.predict(X)
    confidences = model.predict_proba(X).max(axis=1)
    confidences = [round(conf, 2) for conf in confidences]

    return predictions, confidences

def get_prediction_with_keywords(df_review, df_keywords, col_target='review_or_remarks', transform=False):
    """
    Get the prediction with existing processed record and calculate confidence scores.

    Args:
      df_review (DataFrame): A DataFrame containing new tickets.
      df_keywords (DataFrame): A DataFrame containing compiled processed record.
      col_target (str): The name of the column containing the reviews in the df_review DataFrame.

    Returns:
      list: A list of predicted labels for each review in df_review.
    """
    def label_check(text, labels):
        """Check for the presence of labels in a given review text."""
        iter = 0
        check = False
        while ((iter < len(labels)) & (check is False)):
            check = labels[iter] in text
            iter += 1

        if check:
            return labels[iter-1]
        return 'not_found'
    
    def transform_class(tag):
        """Transforming class or label name."""
        tag_transform = re.sub(rf'[{string.punctuation}]', ' ', tag)
        tag_transform = re.sub(r'\s+', ' ', tag_transform)
        tag_transform = tag_transform.lower().strip().replace(' ', '_')
    
        return tag_transform

    keywords = df_keywords.melt()
    keywords = keywords[~(keywords.applymap(lambda x: str(x).strip() == '').any(axis=1))]
    keywords = keywords.dropna(how='all')
    keywords = keywords[keywords['value'].str.len() > 0].reset_index(drop=True)
    keywords['variable'] = "<!>" + keywords['variable'] + "<!>"

    if transform:
        keywords['variable'] = keywords['variable'].apply(transform_class)

    processor = KeywordProcessor(case_sensitive=False)
    for i in range(len(keywords)):
        processor.add_keyword(str(keywords['value'][i]).lower(), keywords['variable'][i])

    all_cat = keywords['variable'].drop_duplicates().map(lambda x: x.lstrip('<!>').rstrip('<!>')).reset_index(drop=True)
    processed = [processor.replace_keywords(df_review[col_target][i].lower()).split("<!>") for i in range(len(df_review))]
    matched_labels = [label_check(fb, all_cat) for fb in processed]

    return matched_labels

def get_prediction_with_record(df_review, df_record, col_target='review_or_remarks', min_occurrence=2):
    """
    Get the prediction with existing processed record and calculate confidence scores.

    Args:
      df_review: A dataframe containing new tickets.
      df_record: A dataframe containing compiled processed record.
      col_target: The name of the column containing the reviews in the df_review dataframe.
      min_occurrence: Minimum occurrence of review_transform from df_record.

    Returns:
      A tuple of two NumPy arrays, 
      where the first array contains the predicted tags and the second array contains the confidence scores.
    """

    df_review[col_target] = df_review[col_target].str.lower()
    df_record['occurrence'] = df_record['occurrence'].astype(int)
    df_record = df_record[df_record['occurrence'] >= min_occurrence]

    merged_df = df_review.merge(df_record, left_on=col_target, right_on='review_recorded', how='left')
    merged_df['tag_recorded'].fillna('not_found', inplace=True)
    merged_df['occurrence'].fillna(0, inplace=True)

    predictions = merged_df['tag_recorded'].to_numpy()
    occurrences = merged_df['occurrence'].to_numpy()

    initial_score = 0.8
    confidences = initial_score + (0.01 * occurrences)
    confidences = np.minimum(1.0, confidences)

    not_found_mask = predictions == 'not_found'
    confidences[not_found_mask] = 0
    confidences = [f"{conf:.2f}" for conf in confidences]

    return predictions, confidences

def multi_model_prediction(df_reviews, col_target, task_id, model_path, model_ids):
    """
    Predict using multiple machine learning models on text data.

    Parameters:
    - df_reviews (DataFrame): A DataFrame containing the text data to be used for predictions.
    - col_target (str): The name of the column in df_reviews that contains the text data.
    - task_id (str): A string specifying the task identifier.
    - model_path (str): The directory path where the trained models and vectorizers are stored.
    - model_ids (list of str): A list of model identifiers to indicate which models to use for predictions.

    Returns:
    - df_preds (DataFrame): A DataFrame containing the original review text and predictions made by each of the specified models. 
      Each model's predictions are stored in separate columns with names like 'pred_model_id'. 
    """

    df_preds = df_reviews[[col_target]].copy()
    for model_id in model_ids:
        print(f'[INFO] Predicting with {model_id}')
        clf = joblib.load(os.path.join(model_path, f'{task_id}_model_latest_{model_id}.joblib'))
        vec = joblib.load(os.path.join(model_path, f'{task_id}_feature_latest_{model_id}.joblib'))
        pred, conf = get_prediction_with_model(df_preds, clf, vec, col_target)
        df_preds[f'pred_{model_id}'] = list(zip(pred, conf))

    return df_preds
    
def ensemble_prediction(df, method):
    """
    Generate ensemble predictions based on the specified method.

    Parameters:
    - df (DataFrame): A DataFrame containing individual model predictions in separate columns.
    - method (str): The ensemble method to be used. Supported methods are 'majority_voting', 'weighted_average', and 'highest_confidence'.

    Returns:
    - df_ensemble (DataFrame): A DataFrame with two columns: 'final_prediction' and 'final_confidence', containing the ensemble prediction and confidence scores.

    Additional Information:
    - This function calculates ensemble predictions for each row in the DataFrame by considering the individual model predictions and their confidence scores.
    - The 'method' parameter determines the ensemble strategy used.
        - 'majority_voting' selects the most commonly predicted class.
        - 'weighted_average' uses the class with the highest average confidence score.
        - 'highest_confidence' selects the class with the highest confidence score.
    """

    pred_cols = [col for col in df.columns if col.startswith("pred_")]
    df_ensemble = pd.DataFrame(columns=["final_prediction", "final_confidence"])

    for i in range(len(df)):
        pred_classes = []
        pred_confidences = []
        for col in pred_cols:
            pred_class, pred_confidence = df.loc[i, col]
            pred_classes.append(pred_class)
            try:
                pred_confidences.append(float(pred_confidence))
            except ValueError:
                pred_confidences.append(0.0)

        if method == "majority_voting":
            final_prediction = max(set(pred_classes), key=pred_classes.count)
            final_confidence = format(pred_confidences[pred_classes.index(final_prediction)], ".2f")
        elif method == "weighted_average":
            if not pred_confidences:
                final_prediction = ''
                final_confidence = '0.00'
            else:
                final_prediction = pred_classes[pred_confidences.index(max(pred_confidences))]
                final_confidence = format(np.mean(pred_confidences), ".2f")
        elif method == "highest_confidence":
            max_confidence_index = pred_confidences.index(max(pred_confidences))
            final_prediction = pred_classes[max_confidence_index]
            final_confidence = format(max(pred_confidences), ".2f")
        else:
            raise ValueError("Invalid method: {}".format(method))

        df_ensemble.loc[i] = [final_prediction, final_confidence]

    return df_ensemble

def get_all_ensemble_prediction(df_preds):
    """
    Perform ensemble predictions using different methods and add the results as new columns to the input DataFrame.

    Parameters:
    - df_preds (DataFrame): A DataFrame containing model predictions to be used for ensemble methods.

    Additional Information:
    - This function calls the 'ensemble_prediction' function with three different ensemble methods: majority voting,
      weighted average, and highest confidence.
    - The ensemble results for each method are added as new columns to the input DataFrame.

    Returns:
    - df_preds (DataFrame): The input DataFrame with added columns for ensemble predictions using different methods.

    Example Usage:
    get_ensemble_prediction(predictions_df)
    """
    ensemble_vot_result_list = list(ensemble_prediction(df_preds, method="majority_voting").itertuples(index=False))
    ensemble_avg_result_list = list(ensemble_prediction(df_preds, method="weighted_average").itertuples(index=False))
    ensemble_con_result_list = list(ensemble_prediction(df_preds, method="highest_confidence").itertuples(index=False))

    df_preds["pred_ensemble_vot"] = ensemble_vot_result_list
    df_preds["pred_ensemble_avg"] = ensemble_avg_result_list
    df_preds["pred_ensemble_con"] = ensemble_con_result_list

    return df_preds

def custom_multi_pred(df_tickets, df_keyword, df_record, classifier, vectorizer, col_target='review_or_remarks'):
    """
    Perform custom multi-prediction based on keyword matching, record lookup, and model prediction.

    Parameters:
        df_tickets (DataFrame): DataFrame containing the ticket data.
        df_keyword (DataFrame): DataFrame containing the keyword data.
        df_record (DataFrame): DataFrame containing the record data.
        classifier: The trained classifier model.
        vectorizer: The trained vectorizer for text data.
        col_target (str): The name of the target column.

    Returns:
        tuple: A tuple containing two lists - pred_final and conf_final.
            - pred_final: A list of the final predicted values.
            - conf_final: A list of the corresponding confidence scores.
    """
    
    pred_final = []
    conf_final = []

    # Keyword Matching
    df_tickets['pred_keyword'] = get_prediction_with_keywords(df_tickets, df_keyword, col_target)

    # Record Lookup
    pred_record, conf_record = get_prediction_with_record(df_tickets, df_record, col_target)
    df_tickets['pred_record'] = pred_record
    df_tickets['conf_record'] = conf_record

    # Model Prediction
    pred_model, conf_model = get_prediction_with_model(df_tickets, classifier, vectorizer, col_target)
    df_tickets['pred_model'] = pred_model
    df_tickets['conf_model'] = conf_model

    for index, row in df_tickets.iterrows():
        pred_model = row['pred_model']
        pred_record = row['pred_record']
        pred_keyword = row['pred_keyword']

        conf_model = pd.to_numeric(row['conf_model'], errors='coerce')
        conf_record = pd.to_numeric(row['conf_record'], errors='coerce')

        # Rule 1: If pred_record is not 'not_found' and conf_model is less than 0.7, use pred_record and conf_record
        if pred_record != 'not_found' and conf_record >= 0.95:
            pred_final.append(pred_record)
            conf_final.append(conf_record)
        else:
            # Rule 2: If pred_keyword and pred_model are the same, add conf_model with +0.2 (maximum 0.99)
            if pred_keyword == pred_model:
                pred_final.append(pred_model)
                conf_final.append(min(0.99, conf_model + 0.2))
            else:
                # Rule 3: If pred_keyword is not 'not_found' and pred_keyword is not same as pred_model, and conf_model is less than 0.4
                if pred_keyword != 'not_found' and pred_keyword != pred_model and conf_model < 0.4:
                    pred_final.append(pred_keyword)
                    conf_final.append(0.8)
                else:
                    # Default: Use pred_model and conf_model
                    pred_final.append(pred_model)
                    conf_final.append(conf_model)

    return pred_final, conf_final

def adhoc_get_models_accuracies(df_preds, cols_pred_identifier, col_actual_tag):
    """
    Calculate accuracies for multiple model predictions.

    Parameters:
    - df_preds (DataFrame): A DataFrame containing the model predictions, where each model's predictions are stored in separate columns.
    - cols_pred_identifier (str): A prefix that identifies the columns containing model predictions.
    - col_actual_tag (str): The name of the column in df_preds that contains the actual target labels for the predictions.

    Returns:
    - accuracy_summary (DataFrame): A DataFrame summarizing the accuracy of each model's predictions. It lists the model column names (identified by cols_pred_identifier) and their respective accuracy values.
    """
     
    def calculate_accuracy(pred_column, actual_column):
        pred_class = [pred[0] for pred in pred_column]
        total_samples = len(actual_column)
        correct_predictions = (pred_class == actual_column).sum()
        accuracy = correct_predictions / total_samples
        return accuracy

    accuracies = {}

    for col in df_preds.columns:
        if col.startswith(cols_pred_identifier):
            accuracy = calculate_accuracy(df_preds[col], df_preds[col_actual_tag])
            accuracies[col] = accuracy

    accuracy_summary = pd.DataFrame.from_dict(accuracies, orient='index', columns=['accuracy'])
    
    return accuracy_summary
