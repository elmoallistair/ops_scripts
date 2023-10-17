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
    df_reviews['review_transform'] = df_reviews['review'].apply(transform_review)
    df_reviews['tag_transform'] = df_reviews['tag'].apply(transform_class)
    df_reviews.dropna(inplace=True)

    value_counts = df_reviews['tag'].value_counts()
    classes = value_counts.index[value_counts >= min_member]
    df_reviews = df_reviews[df_reviews['tag'].isin(classes)]
    df_reviews.reset_index(drop=True, inplace=True)

    return df_reviews

def remove_duplicated_sample(df_reviews):
    df_reviews = df_reviews.copy()
    df_reviews['join'] = df_reviews['tag_transform'] + ' ' + df_reviews['review_transform']
    df_reviews['occurrence'] = df_reviews.groupby('review_transform')['join'].transform('count')

    df_reviews = df_reviews.sort_values(by='occurrence', ascending=False)
    df_reviews = df_reviews.drop_duplicates(subset='review_transform', keep='first')
    df_reviews.reset_index(drop=True, inplace=True)
    
    return df_reviews
