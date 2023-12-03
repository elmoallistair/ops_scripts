import re
import string

def transform_review(raw_review):
    raw_review = str(raw_review) 
    review_transform = re.sub(f"[{string.punctuation}]", ' ', raw_review)
    review_transform = re.sub('\n', ' ', review_transform)
    review_transform = re.sub(r'\s+', ' ', review_transform).strip()
    review_transform = review_transform.encode('ascii', 'ignore').decode('ascii')
    review_transform = review_transform.lower()
    
    return review_transform

def transform_class(tag):
    tag_transform = re.sub(rf'[{string.punctuation}]', ' ', tag)
    tag_transform = re.sub(r'\s+', ' ', tag_transform)
    tag_transform = tag_transform.lower().strip().replace(' ', '_')
    
    return tag_transform

def clean_sample_data(df_reviews, min_member=2):
    df_reviews['review_recorded'] = df_reviews['review'].apply(transform_review)
    df_reviews['tag_recorded'] = df_reviews['tag'].apply(transform_class)
    df_reviews.dropna(inplace=True)

    value_counts = df_reviews['tag'].value_counts()
    classes = value_counts.index[value_counts >= min_member]
    df_reviews = df_reviews[df_reviews['tag'].isin(classes)]
    df_reviews.reset_index(drop=True, inplace=True)

    return df_reviews

def remove_duplicated_sample(df_reviews, prioritize_latest=True):
    df_reviews = df_reviews.copy()

    if prioritize_latest:
        df_reviews.sort_values(by='date_append', ascending=False, inplace=True)
        
    df_reviews['join'] = df_reviews['tag_recorded'] + ' ' + df_reviews['review_recorded']
    df_reviews['occurrence'] = df_reviews.groupby('review_recorded')['join'].transform('count')

    df_reviews.sort_values(by=['review_recorded', 'occurrence'], ascending=[True, False], inplace=True)
    idx_first_occurrence = df_reviews.groupby('review_recorded').head(1).index
    df_reviews = df_reviews.loc[idx_first_occurrence]
    df_reviews.reset_index(drop=True, inplace=True)
    
    return df_reviews

def filter_class(df, exc_class, target, min_population=2):
    """
    Filter a DataFrame based on the values in the specified target column, excluding specified classes and retaining tags with a minimum population count.

    Parameters:
    - df: The input DataFrame to be filtered.
    - exc_class: A list of classes to be excluded from the DataFrame.
    - target: The name of the column containing the tags/classes.
    - min_population: The minimum population count for a class to be retained. Tags with a population count less than this value will be excluded.

    Returns:
    - df_filtered: The filtered DataFrame that excludes specified classes and retains tags with the minimum population.
    """
    
    df_filtered = df[~df[target].isin(exc_class)].copy()
    tag_counts = df_filtered[target].value_counts()
    tags_to_keep = tag_counts[tag_counts >= min_population].index
    df_filtered = df_filtered[df_filtered[target].isin(tags_to_keep)]
    return df_filtered
