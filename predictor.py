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

    merged_df = df_review.merge(df_record, left_on=col_name, right_on='review_transform', how='left')
    merged_df['tag_transform'].fillna(np.nan, inplace=True)
    merged_df['occurrence'].fillna(0, inplace=True)
    predictions = merged_df['tag_transform'].to_numpy()
    occurrences = merged_df['occurrence'].to_numpy()

    initial_score = 0.8
    confidences = initial_score + (0.02 * occurrences)
    confidences = np.minimum(1.0, confidences)
    confidences[pd.isna(predictions)] = np.nan

    return predictions, confidences
