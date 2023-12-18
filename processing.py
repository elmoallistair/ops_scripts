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

def validate_tickets(df_tickets, territories=None, verticals=None, drop=False):
    def is_valid_booking_code(code):
        valid_prefixes = ['A-', 'IN-', 'SD-', 'MS-']
        valid_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-")
        return any(code.startswith(prefix) for prefix in valid_prefixes) and all(c in valid_chars for c in code)

    invalid_values = ['#N/A', '#REF!', '#ERROR!', '#NAME?', '#N/A']
    df_tickets = df_tickets.replace(invalid_values, float('nan'))

    # Validate driver_id
    non_numeric_driver_ids = df_tickets[df_tickets['driver_id'].astype(str).str.contains(r'\D', na=False)]
    df_tickets['driver_id'] = pd.to_numeric(df_tickets['driver_id'], errors='coerce')
    invalid_driver_ids = df_tickets[(df_tickets['driver_id'] < 50000) | (df_tickets['driver_id'] > 100000000)]
    invalid_driver_ids = pd.concat([invalid_driver_ids, non_numeric_driver_ids])

    if len(invalid_driver_ids) > 0:
        if drop:
            df_tickets = df_tickets.drop(invalid_driver_ids.index)
            print(f'[REJECTED] {len(invalid_driver_ids)} ticket(s) has invalid driver_id', end=': ')
        else:
            print(f'[WARNING] {len(invalid_driver_ids)} ticket(s) has invalid driver_id', end=': ')
        print(invalid_driver_ids['driver_id'].values)

    # Validate booking_code
    invalid_booking_codes = df_tickets[~df_tickets['booking_code'].apply(is_valid_booking_code)]
    if len(invalid_booking_codes) > 0:
        if drop:
            df_tickets = df_tickets.drop(invalid_booking_codes.index)
            print(f'[REJECTED] {len(invalid_booking_codes)} ticket(s) has invalid booking_code', end=': ')
        else:
            print(f'[WARNING] {len(invalid_booking_codes)} ticket(s) has invalid booking_code', end=': ')
        print(invalid_booking_codes['booking_code'].values)

    # Validate territories
    if territories is not None:
        unlisted_territories = df_tickets[~df_tickets['territory'].isin(territories)]
        if len(unlisted_territories) > 0:
            if drop:
                print(f'[REJECTED] {len(unlisted_territories)} ticket(s) has unlisted city_name', end=': ')
            else:
                print(f'[WARNING] {len(unlisted_territories)} ticket(s) has unlisted city_name', end=': ')
            print(unlisted_territories['territory'].values)

    # Validate vertical
    if verticals is not None:
        unlisted_verticals = df_tickets[~df_tickets['vertical'].isin(verticals)]
        if len(unlisted_verticals) > 0:
            if drop:
                df_tickets = df_tickets[df_tickets['vertical'].isin(verticals)]
                print(f'[REJECTED] {len(unlisted_verticals)} ticket(s) has unlisted vertical', end=': ')
            else:
                print(f'[WARNING] {len(unlisted_verticals)} ticket(s) has unlisted vertical', end=': ')
            print(unlisted_verticals['vertical'].values)

    # Reset index
    df_tickets = df_tickets.reset_index(drop=True)

    return df_tickets
