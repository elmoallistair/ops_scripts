import re
import unicodedata
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def text_preprocessing(text, keep_punctuations=True):
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = ''.join(char for char in text if unicodedata.category(char)[0] != 'S')
    text = re.sub(r'\s*([.,?!])\s*', r'\1 ', text)
    text = text.replace('\n', '. ')

    if not keep_punctuations:
        text = re.sub(r'[^\w\s]', '', text)

    return text

def lookup_by_reference(df_tickets, df_reference, reference_col):
    merged_df = df_tickets.merge(df_reference[['driver_id', reference_col]], on='driver_id', how='left', suffixes=('_tickets', '_reference'))
    merged_df[reference_col + '_tickets'].fillna(merged_df[reference_col + '_reference'], inplace=True)
    merged_df.rename(columns={reference_col + '_tickets': reference_col}, inplace=True)
    merged_df.drop(reference_col + '_reference', axis=1, inplace=True)

    return merged_df

def rename_columns_with_template(df, df_rename):
    renamed_columns = []
    for col in df.columns:
        col = col.lower().replace(' ', '_')
        for _, row in df_rename.iterrows():
            variants = row['variant'].split('|')
            if col in variants:
                renamed_columns.append(row['origin'])
                break
        else:
            renamed_columns.append(col)
    df.columns = renamed_columns
    return df

def order_column_by_template(dataframe, cols_lst):
    template = pd.Series(cols_lst)[pd.Series(cols_lst) != ''].dropna()
    dataframe_temp = pd.DataFrame(columns=template)

    for col in template:
        if col in dataframe.columns:
            dataframe_temp[col] = dataframe[col]
        else:
            dataframe_temp[col] = np.nan

    return dataframe_temp

def remove_short_reviews(df_reviews, column_name, min_word=2, min_letter=5):
    df_reviews['word_count'] = df_reviews[column_name].str.split().apply(len)
    df_reviews['letter_length'] = df_reviews[column_name].str.len()

    df_reviews_clean = df_reviews[(df_reviews['word_count'] >= min_word) & (df_reviews['letter_length'] >= min_letter)].copy()
    df_reviews_clean.dropna(subset=[column_name], inplace=True)
    df_reviews_clean.drop(columns=['word_count', 'letter_length'], inplace=True)

    removed_count = len(df_reviews) - len(df_reviews_clean)
    if removed_count > 0:
        print(f'[INFO] Removed {removed_count} short reviews from dataset')

    return df_reviews_clean

def create_booking_metadata(row):
    metadata = {
        'bc': 1 if row['is_batching'] else 0,
        'pf': 1 if pd.notna(row['parking_fee']) else 0,
        'pl': int(max(0, row['pick_late'] // 10)) if pd.notna(row['pick_late']) else 0,
        'dl': int(max(0, row['drop_late'] // 10)) if pd.notna(row['drop_late']) else 0,
        'pu': ' '.join(row['pickup_keywords'].split()[:2]) if pd.notna(row['pickup_keywords']) else 'null',
        'do': ' '.join(row['dropoff_keywords'].split()[:2]) if pd.notna(row['dropoff_keywords']) else 'null'
    }
    return metadata

def apply_metadata_to_dataframe(df, metadata):
    for column, value in metadata.items():
        if column == 'lambdas':
            for transformation in value:
                col_name = transformation['name']
                function_str = transformation['function']
                df[col_name] = eval(function_str)
        else:
            df[column] = value

    return df

def detect_suspicious_pax(df):
    excluded_predictions = ['mex_related', 'product_related', 'unclear']
    count_mask = ~df['prediction'].isin(excluded_predictions)

    df['pax_count'] = df[count_mask].groupby('passenger_id')['passenger_id'].transform('count')
    df['identifier_count'] = df[count_mask].groupby('identifier')['identifier'].transform('count')

    suspicious_condition = (df['pax_count'] > 2) | (df['identifier_count'] > 1)
    df.loc[suspicious_condition, 'prediction'] = 'Suspicious'

    return df

def get_wheel_count(row):
    fleet_name = row['fleet_name'].lower()
    return '4W' if fleet_name.startswith(('uni', 'tpi', 'gc')) else '2W'
        
def fix_taxi_types(text):
    text = text.replace("__", "::")
    text = text.replace("_", "")
    text = re.sub(r'\bgrab', 'Grab', text)
    text = re.sub(r'(Grab|::)([a-z])', lambda x: x.group(1) + x.group(2).upper(), text)
    return text

def get_taxi_type_simple(text):
    match = re.search(r'[^: ]+', text)
    if match:
        return match.group()
    else:
        return text

def concat_column_values(series):
    all_numeric = all(pd.to_numeric(series, errors='coerce').notnull())
    series = series.astype(str)

    if all_numeric:
        result = ', '.join(series)
    else:
        result = ', '.join(f"'{value}'" for value in series)

    return result

def validate_tickets(df_tickets):
    def is_valid_booking_code(code):
        return any(code.startswith(prefix) for prefix in valid_prefixes) and all(c in valid_chars for c in code)

    valid_prefixes = ['A-', 'IN-', 'SD-', 'MS-']
    valid_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-")

    df_valid = df_tickets.copy()
    df_valid[['driver_id', 'booking_code']] = df_valid[['driver_id', 'booking_code']].astype(str)
    df_valid = df_valid[(df_valid['driver_id'].str.strip() != '') & (df_valid['booking_code'].str.strip() != '')]

    df_valid['driver_id'] = pd.to_numeric(df_valid['driver_id'], errors='coerce')
    df_valid = df_valid[(df_valid['driver_id'] > 0) & df_valid['driver_id'].notna()] 
    df_valid = df_valid[df_valid['booking_code'].apply(is_valid_booking_code)]

    return df_valid

