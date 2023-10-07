import re
import string
import warnings
import unicodedata
import pandas as pd
import numpy as np

def lookup_value(df_source_key, df_target, old_value, new_value):
    matching_row = df_target[df_target[old_value] == df_source_key]
    if len(matching_row) > 0:
        return matching_row.iloc[0][new_value]
    else:
        return None
  
def rename_columns_with_template(column, df_rename):
    column = column.lower().replace(' ', '_')
    for _, row in df_rename.iterrows():
        variants = row['Variant'].split('|')
        if column in variants:
            return row['Origin']
    return column

def order_column_by_template(dataframe, cols_lst):
    template = pd.Series(cols_lst)[pd.Series(cols_lst) != ''].dropna()
    dataframe_temp = pd.DataFrame(columns=template)

    for col in template:
        if col in dataframe.columns:
            dataframe_temp[col] = dataframe[col]
        else:
            dataframe_temp[col] = np.nan

    return dataframe_temp

def text_preprocessing(text, keep_punctuations=True):
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = ''.join(char for char in text if unicodedata.category(char)[0] != 'S')
    text = re.sub(r'\s*([.,?!])\s*', r'\1 ', text)
    text = text.replace('\n', '. ')

    if not keep_punctuations:
        text = re.sub(r'[^\w\s]', '', text)

    return text

def remove_short_reviews(df_reviews, column_name, min_word_count=2, min_letter_length=10):
    df_reviews['word_count'] = df_reviews[column_name].str.split().apply(len)
    df_reviews['letter_length'] = df_reviews[column_name].str.len()

    df_reviews_clean = df_reviews[(df_reviews['word_count'] >= min_word_count) & (df_reviews['letter_length'] >= min_letter_length)].copy()
    df_reviews_clean.dropna(subset=[column_name], inplace=True)
    df_reviews_clean.drop(columns=['word_count', 'letter_length'], inplace=True)
    print('Removed {len(df_reviews) - len(df_reviews_clean) short reviews from dataset}')

    return df_reviews_clean
