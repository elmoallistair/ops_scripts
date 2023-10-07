import re
import string
import pandas as pd

def clean_review_text(raw_review):
    raw_review = str(raw_review) 
    review_transform = re.sub(f"[{string.punctuation}]", ' ', raw_review)
    review_transform = re.sub('\n', ' ', review_transform)
    review_transform = re.sub(r'\s+', ' ', review_transform).strip()
    review_transform = review_transform.encode('ascii', 'ignore').decode('ascii')
    review_transform = review_transform.lower()
    return review_transform

def transform_class(tag):
    tag = re.sub(rf'[{string.punctuation}]', ' ', tag)
    tag = re.sub(r'\s+', ' ', tag)
    tag = tag.lower().strip().replace(' ', '_')
    return tag

def clean_sample_data(df_reviews):
    df_reviews['review_transform'] = df_reviews['review'].apply(clean_review_text)
    df_reviews['tag_transform'] = df_reviews['tag'].apply(transform_class)
    df_reviews.dropna(inplace=True)

    df_reviews['word_len'] = df_reviews['review'].apply(lambda x: len(x.split()))
    df_reviews['string_len'] = df_reviews['review'].apply(lambda x: len(x))

    value_counts = df_reviews['tag'].value_counts()
    df_reviews = df_reviews[df_reviews['tag'].isin(value_counts.index[value_counts >= 10])]

    return df_reviews

def remove_duplicated_sample(df_reviews):
    df_reviews['join'] = df_reviews['tag'] + ' ' + df_reviews['review']
    df_reviews['occurrence'] = df_reviews.groupby('review_transform')['join'].transform('count')
    df_reviews = df_reviews.sort_values(by='occurrence', ascending=False)
    df_reviews = df_reviews.drop_duplicates(subset='review_transform', keep='first')
    df_reviews.reset_index(drop=True, inplace=True)
    
    return df_reviews[['review_transform', 'tag_transform', 'occurrence', 'word_len', 'string_len']]
