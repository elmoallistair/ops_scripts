import re
import json
import unicodedata
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def create_identifier(df, columns, separator=', '):
    df['identifier'] = df[columns].astype(str).apply(separator.join, axis=1)
    return df

def lookup_by_reference(df_tickets, df_reference, reference_col, match_col):
    reference_dict = df_reference.set_index(match_col)[reference_col].to_dict()
    df_tickets[reference_col] = df_tickets[match_col].map(reference_dict)
    return df_tickets

def text_preprocessing(df, column_target='review_or_remarks', keep_punctuations=True):
    def preprocess_text(text):
        text = re.sub(r'\s+', ' ', str(text).lower().strip())
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'S')
        text = re.sub(r'\s*([.,?!])\s*', r'\1 ', text.replace('\n', '. '))
        return re.sub(r'[^\w\s]', '', text) if not keep_punctuations else text

    df[column_target] = df[column_target].astype(str).apply(preprocess_text)
    return df

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

def order_column_by_template(dataframe, cols_lst, nan_value=''):
    template = [col for col in cols_lst if col != '']
    dataframe = dataframe.reindex(columns=template)
    dataframe = dataframe.fillna(nan_value)
    
    return dataframe

def remove_short_reviews(df_reviews, column_name='review_or_remarks', min_word=2, min_letter=5):
    df_reviews = df_reviews[df_reviews[column_name].str.split().str.len() >= min_word]
    df_reviews = df_reviews[df_reviews[column_name].str.len() >= min_letter]
    df_reviews.dropna(subset=[column_name], inplace=True)

    removed_count = len(df_reviews) - len(df_reviews)
    if removed_count > 0:
        print(f'[INFO] Removed {removed_count} short reviews from dataset')

    df_reviews = df_reviews.reset_index(drop=True)

    return df_reviews

def create_booking_metadata(df):
    def metadata(row):
        data = {
            'bc': row['is_batching'],
            'pf': row['parking_fee'],
            'pl': row['pick_late'],
            'dl': row['drop_late'],
            'pu': row['poi_pickup'],
            'do': row['poi_dropoff']
        }
        return json.dumps(data)

    df['booking_meta'] = df.apply(metadata, axis=1)
    return df

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

    df = create_identifier(df, ['passenger_id', 'review_or_remarks'])
    df['pax_count'] = df[count_mask].groupby('passenger_id')['passenger_id'].transform('count')
    df['identifier_count'] = df[count_mask].groupby('identifier')['identifier'].transform('count')

    suspicious_condition = (df['pax_count'] > 2) | (df['identifier_count'] > 1)
    df.loc[suspicious_condition, 'prediction'] = 'suspicious'

    return df

def get_wheel_count(row):
    fleet_name = row['fleet_name'].lower()
    return '4W' if fleet_name.startswith(('uni', 'tpi', 'gc')) else '2W'
        
def fix_taxi_type(df, col_name='taxi_type'):
    df[col_name] = df[col_name].str.replace('__', '::')
    df[col_name] = df[col_name].str.replace('_', '')
    df[col_name] = df[col_name].str.replace(r'\bgrab', 'Grab', regex=True)
    df[col_name] = df[col_name].str.replace(r'(Grab|::)([a-z])', lambda x: x.group(1) + x.group(2).upper(), regex=True)
    
    return df

def fix_city_name(df, col_name='city_name'):
    """Converts 'country__area__city' format to 'city' only"""
    df[col_name] = df[col_name].apply(lambda x: x.split('__')[-1].capitalize())
    
    return df
    
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

def append_zendesk_ticket_id(df_tickets, col_target='review_or_remarks', col_id='ticket_id'):
    df_tickets[col_target] = 'Zendesk Ticket ID ' + df_tickets[col_id].astype(str)
    
    return df_tickets
    
def validate_tickets(df_tickets, territories=None, verticals=None, drop=False, reject=False):
    def is_valid_booking_code(code):
        valid_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-')
        return all(c in valid_chars for c in code)

    flag = False
    invalid_values = ['#N/A', 'NaN', '#REF!', '#ERROR!', '#NAME?', '#N/A']
    df_tickets = df_tickets.replace(invalid_values, np.nan)
    
    # Validate driver_id
    non_numeric_driver_ids = df_tickets[df_tickets['driver_id'].astype(str).str.contains(r'\D', na=False)]
    df_tickets['driver_id'] = pd.to_numeric(df_tickets['driver_id'], errors='coerce')
    invalid_driver_ids = df_tickets[(df_tickets['driver_id'] < 50000) | (df_tickets['driver_id'] > 100000000)]
    invalid_driver_ids = pd.concat([invalid_driver_ids, non_numeric_driver_ids])
    
    if len(invalid_driver_ids) > 0:
        if drop:
            df_tickets = df_tickets.drop(invalid_driver_ids.index)
            flag = True

        print(f'[WARNING] {len(invalid_driver_ids)} ticket(s) has invalid driver_id')

    # Validate booking_code
    invalid_booking_codes = df_tickets[~df_tickets['booking_code'].apply(is_valid_booking_code)]
    if len(invalid_booking_codes) > 0:
        if drop:
            df_tickets = df_tickets.drop(invalid_booking_codes.index)
            flag = True
        
        print(f'[WARNING] {len(invalid_booking_codes)} ticket(s) has invalid booking_code')

    # Validate territories
    if territories is not None:
        unlisted_territories = df_tickets[~df_tickets['territory'].isin(territories)]
        if len(unlisted_territories) > 0:
            if drop:
                df_tickets = df_tickets[df_tickets['territory'].isin(territories)]
                flag = True
            
            print(f'[WARNING] {len(unlisted_territories)} ticket(s) has unlisted territory')

    # Validate vertical
    if verticals is not None:
        unlisted_verticals = df_tickets[~df_tickets['vertical'].isin(verticals)]
        if len(unlisted_verticals) > 0:
            if drop:
                df_tickets = df_tickets[df_tickets['vertical'].isin(verticals)]
                flag = True
        
            print(f'[WARNING] {len(unlisted_verticals)} ticket(s) has unlisted vertical')
            print(unlisted_verticals['vertical'].values)

    # Reset index
    df_tickets = df_tickets.reset_index(drop=True)

    if reject and flag:
        print('[REJECTED] Please fix the issues above')
        return None

    return df_tickets
