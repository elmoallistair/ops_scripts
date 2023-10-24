import pandas as pd
import numpy as np

def get_prediction_with_model(df_review, model, vectorizer, col_name):
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

    X = vectorizer.transform(df_review[col_name])
    predictions = model.predict(X)
    confidences = model.predict_proba(X).max(axis=1)
    confidences = [round(conf, 2) for conf in confidences]

    return predictions, confidences

def get_prediction_with_record(df_review, df_record, col_name, min_occurrence=2):
    """
    Get the prediction with existing processed record and calculate confidence scores.

    Args:
      df_review: A dataframe containing new tickets.
      df_record: A dataframe containing compiled processed record.
      col_name: The name of the column containing the reviews in the df_review dataframe.
      min_occurrence: Minimum occurrence of review_transform from df_record.

    Returns:
      A tuple of two NumPy arrays, 
      where the first array contains the predicted tags and the second array contains the confidence scores.
    """

    df_review[col_name] = df_review[col_name].str.lower()
    df_record['occurrence'] = df_record['occurrence'].astype(int)
    df_record = df_record[df_record['occurrence'] >= min_occurrence]

    merged_df = df_review.merge(df_record, left_on=col_name, right_on='review_recorded', how='left')
    merged_df['tag_recorded'].fillna(np.nan, inplace=True)
    merged_df['occurrence'].fillna(0, inplace=True)
    predictions = merged_df['tag_recorded'].to_numpy()
    occurrences = merged_df['occurrence'].to_numpy()

    initial_score = 0.8
    confidences = initial_score + (0.02 * occurrences)
    confidences = np.minimum(1.0, confidences)
    confidences[pd.isna(predictions)] = 0

    # Format confidences with two decimal places
    confidences = [f"{conf:.2f}" for conf in confidences]

    return predictions, confidences

def ensemble_prediction(df, method):
    """
    Combining multiple model prediction, based on methods.

    Args:
        df: A Pandas DataFrame containing the data to be predicted.
        method: The method to use for ensemble prediction. Can be either "majority_voting", "weighted_average",
                or "highest_confidence".

    Returns:
        A Pandas DataFrame containing the ensemble predictions.
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

