import re
import unicodedata
import numpy as np
import pandas as pd
import datetime as dt
import google_connect as pconnect
import presto_connect as pconnect

def get_n_last_date_range(n):
    today_date = dt.date.today()
    date_format = "%Y-%m-%d"
    if n == 0:
        date_start = date_end = today_date.strftime(date_format)
    elif n == 1:
        yesterday_date = today_date - dt.timedelta(days=1)
        date_start = yesterday_date.strftime(date_format)
        date_end = today_date.strftime(date_format)
    else: 
        date_end = today_date.strftime(date_format)
        date_start = (today_date - dt.timedelta(days=n)).strftime(date_format)

    return date_start, date_end

def get_date_range(start_date, end_date):
    date_list = []
    current_date = dt.datetime.strptime(start_date, '%Y-%m-%d')

    while current_date <= dt.datetime.strptime(end_date, '%Y-%m-%d'):
        date_list.append(current_date.strftime('%Y-%m-%d'))
        current_date += dt.timedelta(days=1)

    return date_list

def remove_empty_rows(row):
    return any(isinstance(value, str) and value.strip() == '' or pd.isna(value) for value in row)
    
def lookup_value(df_source_key, df_target, old_value, new_value):
    matching_row = df_target[df_target[old_value] == df_source_key]
    if len(matching_row) > 0:
        return matching_row.iloc[0][new_value]
    else:
        return None

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

def correct_prediction_label(label, lookup_df, lookup_column='transform', result_column='origin'):
    match = lookup_df[lookup_df[lookup_column] == label]
    if not match.empty:
        return match.iloc[0][result_column]
    else:
        return label

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
    print(f'Removed {len(df_reviews) - len(df_reviews_clean)} short reviews from dataset')

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

def get_suspicious_reviews(df):
    class_exclude = ['Unclear', 'Mex Related', 'Product Related']
    df['is_duplicated'] = df.duplicated(subset=['passenger_id', 'review_or_remarks'], keep=False)
    df.loc[df['is_duplicated'] & ~df['prediction'].isin(class_exclude), 'prediction'] = 'Suspicious'
    df.loc[df['prediction'] == 'Suspicious', 'conf_score'] = 0.99
    df.drop('is_duplicated', axis=1, inplace=True)

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

def lookup_value_by_reference(df_source, df_reference, source_column, reference_column, key_id):
    result_df = df_source.copy()

    result_df[source_column] = result_df[source_column].astype(str)
    df_reference[reference_column] = df_reference[reference_column].astype(str)
    result_df[key_id] = result_df[key_id].astype(str)
    df_reference[key_id] = df_reference[key_id].astype(str)

    for index, row in result_df.iterrows():
        key_value = row[key_id]
        reference_row = df_reference[df_reference[key_id] == key_value]

        if not reference_row.empty:
            result_df.at[index, source_column] = reference_row.iloc[0][reference_column]

    return result_df

def retrieve_dax_info(driver_ids):
    df_queries = gconnect.read_data_from_sheet(client, sheet_config, sheet_name='queries')
    query_dax_info = df_queries['get_driver_info'][0]
    query_dax_info = query_dax_info.format(driver_ids=driver_ids)
    df_driver_infos = pconnect.execute_presto_query(query_dax_info, dec('cqq+`ihj+m`wl|dkqj', 5), dec('Gjojkb75<&', 5))
    return df_driver_infos
    
def append_driver_info(df_tickets, append_col):
    identifiers = concat_column_values(df_tickets['driver_id'])
    driver_infos = retrieve_dax_info(driver_ids)

    df_tickets['driver_id'] = df_tickets['driver_id'].astype(int)
    df_tickets[append_col] = ''
    df_tickets[append_col] = lookup_value_by_reference(df_tickets, driver_infos, append_col, append_col, 'driver_id')

def append_metadata_to_dataframe(df, metadata):
    for column, value in metadata.items():
        if column == 'lambdas':
            for transformation in value:
                col_name = transformation['name']
                function_str = transformation['function']
                df[col_name] = eval(function_str)
        else:
            df[column] = value
            
# ========== BAU PROCESSING ==========

def get_treatment_name(vertical, category=None):
    treatment_mapping = {
        'd': 'DELIHCLGF',
        't': 'DELIHCLGM',
        's': 'DELIHCLGE'
    }
    
    return treatment_mapping.get(vertical[-1], '')

def process_pax_rating(df_prt_raw, model, feature, mappings):
    cols_order, cols_rename = mappings.values()
    print('Cleaning reviews...')
    df_prt = df_prt_raw.copy()
    df_prt = rename_columns_with_template(df_prt, cols_rename)
    df_prt['review_or_remarks'] = df_prt['review_or_remarks'].apply(lambda x: text_preprocessing(x, keep_punctuations=False))

    df_prt['is_batching'] = df_prt['is_batching'].notna()
    df_prt['pax_count'] = df_prt.groupby('passenger_id')['review_or_remarks'].transform('count')
    df_prt['booking_meta'] = df_prt.apply(create_booking_metadata, axis=1)

    df_prt = remove_short_reviews(df_prt, column_name="review_or_remarks", min_word_count=2)
    df_prt = order_column_by_template(df_prt, cols_order['prt'])
    df_prt.sort_values(['date_local'], inplace=True)

    df_prt = get_suspicious_reviews(df_prt)
    df_prt['source'] = 'Comments'

    return df_prt

def process_chat(df_chat, mappings, identifier):
    cols_order, cols_rename = mappings.values()
    df_chat = rename_columns_with_template(df_chat, cols_rename)
    df_chat['review_or_remarks'] = df_chat['review_or_remarks'].apply(lambda x: text_preprocessing(x))
    df_chat = remove_short_reviews(df_chat, column_name="review_or_remarks", min_word_count=2)
    df_chat = order_column_by_template(df_chat, cols_order[identifier])
    df_chat.drop_duplicates(subset=['booking_code'], inplace=True)
    df_chat.sort_values(['review_or_remarks'], inplace=True)
    df_chat.reset_index(drop=True, inplace=True)
    df_chat['date_local'] = pd.to_datetime(df_chat['date_local']).dt.date
    df_chat['remarks'] = 'Valid'
    df_chat['source'] = 'Chats'

    return df_chat

def process_zendesk(df_zendesk, gs_tags, mappings):
    def get_zendesk_subdispo(reference, df_gs_tag):
        matching_row = df_gs_tag[df_gs_tag['id_gs_coc'] == reference]
        if not matching_row.empty:
            return matching_row['coc_subdispo'].values[0]
        else:
            return None

    cols_order, cols_rename = mappings.values()
    df_zendesk = rename_columns_with_template(df_zendesk, cols_rename)
    df_zendesk = df_zendesk[df_zendesk['id_gs_coc'].notnull() & (df_zendesk['id_gs_coc'] != '')]
    df_zendesk['date_local'] = pd.to_datetime(df_zendesk['date_local']).dt.date
    df_zendesk['sub_disposition'] = df_zendesk['id_gs_coc'].apply(get_zendesk_subdispo, df_gs_tag=gs_tags)
    df_zendesk['review_or_remarks'] = 'Zendesk Ticket ID ' + df_zendesk['ticket_id'].astype(str)

    df_zendesk = order_column_by_template(df_zendesk, cols_order['coc'])
    df_zendesk['remarks'] = 'Valid'
    df_zendesk['source'] = 'Zendesk'
    
    return df_zendesk

def validate_tickets(dataframe):
    dataframe['driver_id'] = dataframe['driver_id'].replace('', np.nan)

    driver_id_conditions = [
        pd.to_numeric(dataframe['driver_id'], errors='coerce').notnull(),
        (dataframe['driver_id'].str.len() >= 5) & (dataframe['driver_id'].str.len() <= 15)
    ]

    booking_code_conditions = [
        dataframe['booking_code'].str.match('^(A-|IN-|SD-|MS-)'),
        (dataframe['booking_code'].str.lower() != 'na'),
        ~dataframe['booking_code'].str.contains('/')
    ]

    valid_tickets = dataframe[np.all(driver_id_conditions, axis=0) & np.all(booking_code_conditions, axis=0)]

    return valid_tickets

def process_cancellation(df_hcl, mappings):
    map_rename = {
        'date_id':'date_local',
        'taxi_type_simple':'vertical',
        'total_allocated':'allocated_booking_count', 
        'total_cancelled_driver':'cancelled_booking_count',
        'cancel_rate_percentage':'cancellation_rate'}

    cols_order, _ = mappings.values()
    df_hcl.rename(columns=map_rename, inplace=True)
    df_hcl['driver_id'] = df_hcl['driver_id'].astype(str)
    df_hcl['identifier'] = df_hcl['driver_id'] + df_hcl['vertical']
    df_hcl['wheels'] = '2'
    df_hcl['category'] = 'Driver Cancellation'
    df_hcl['root_cause'] = 'High Cancellation'
    df_hcl['sub_category'] = 'High Cancellation'
    df_hcl['category_level'] = 'Low'
    df_hcl['final_treatment'] = 'Warning'   # Subject to Change
    df_hcl['origin_treatment'] = 'Warning'  # Subject to Change
    df_hcl['cancellation_rate'] = df_hcl['cancellation_rate'].astype(float) / 100.0
    df_hcl['treatment_name'] = df_hcl['vertical'].apply(get_treatment_name, category='high_cancellation')
    df_hcl = order_column_by_template(df_hcl, cols_order['high_cancellation'])
    return df_hcl

def process_gsid_improper(df_gsid, mappings):
    def get_gsid_subdispo(vertical):
        last_letter = vertical[-1]
        if last_letter == 's':
            return 'Driver Early Completed'
        return 'Not confirm the delivery address'

    cols_order, cols_rename = mappings.values()
    df_gsid = rename_columns_with_template(df_gsid, cols_rename)
    df_gsid = validate_tickets(df_gsid)
    df_gsid = df_gsid[df_gsid['is_daxearlycompleted'] == 'True']

    df_gsid['taxi_type'] = df_gsid['taxi_type'].apply(fix_taxi_types)
    df_gsid['vertical'] = df_gsid['taxi_type'].str.split('::').str[0]
    df_gsid['disposition'] = 'Enquiry::Improper Job Handling'
    df_gsid['review_or_remarks'] = df_gsid['ticket_id'].apply(lambda x: f'Zendesk Ticket ID {x}')
    df_gsid['sub_disposition'] = df_gsid['vertical'].apply(get_gsid_subdispo)

    df_gsid['cleaned_booking_code'] = df_gsid['booking_code'].str.strip().str.lower()
    df_gsid.drop_duplicates(subset='cleaned_booking_code', keep='first', inplace=True)
    df_gsid.drop_duplicates(subset=['driver_id', 'vertical'], keep='first', inplace=True)

    df_gsid = order_column_by_template(df_gsid, cols_order['coc'])
    df_gsid['remarks'] = 'Valid'
    df_gsid['source'] = 'Zendesk'
    return df_gsid
