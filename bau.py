import striking
import processing
import predictor
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def split_tickets(df_tickets, cols_order, identifier):
    df_tickets_coc = df_tickets.drop_duplicates('identifier')
    df_tickets_coc = processing.order_column_by_template(df_tickets_coc, cols_order['coc'])
    df_tickets_docs = processing.order_column_by_template(df_tickets, cols_order[identifier])

    return df_tickets_coc, df_tickets_docs

def postprocess_tickets(df_tickets, sort_by=['date_local', 'driver_id'], identifier='booking_code', remove_duplicate=None):
    if remove_duplicate is not None:
        df_tickets.drop_duplicates(subset=remove_duplicate, inplace=True)
    
    df_tickets.sort_values(sort_by, inplace=True)
    df_tickets.reset_index(drop=True, inplace=True)

    print(f'[SUCCESS] Successfully processed {df_tickets[identifier].nunique()} tickets')
    
    return df_tickets

def process_pax_rating(df_prt_raw, model, feature, metadata, mappings):
    cols_order, cols_rename = mappings.values()

    df_prt = df_prt_raw.copy()
    df_prt = processing.rename_columns_with_template(df_prt, cols_rename)

    df_prt['is_batching'] = df_prt['is_batching'].notna()
    df_prt['booking_meta'] = df_prt.apply(processing.create_booking_metadata, axis=1)
    df_prt['review_or_remarks'] = df_prt['review_or_remarks'].apply(
        lambda x: processing.text_preprocessing(x, keep_punctuations=False))

    pred, conf = predictor.get_prediction_with_model(df_prt, model, feature, 'review_or_remarks')
    df_prt['prediction'] = pred
    df_prt['conf_score'] = conf

    df_prt = processing.apply_metadata_to_dataframe(df_prt, metadata)
    df_prt = processing.detect_suspicious_pax(df_prt)
    df_prt = processing.remove_short_reviews(df_prt, column_name="review_or_remarks", min_word_count=2)
    df_prt = processing.order_column_by_template(df_prt, cols_order['prt'])
    df_prt = postprocess_tickets(df_prt, sort_by=['date_local', 'prediction', 'conf_score'])

    return df_prt

def process_chat(df_chat_raw, mappings, metadata):
    cols_order, cols_rename = mappings.values()

    df_chat = df_chat_raw.copy()
    df_chat = processing.rename_columns_with_template(df_chat, cols_rename)
    df_chat = processing.validate_tickets(df_chat)
    df_chat = processing.apply_metadata_to_dataframe(df_chat, metadata)
    df_chat = processing.remove_short_reviews(df_chat, column_name='review_or_remarks', min_word_count=3)
    df_chat = processing.order_column_by_template(df_chat, cols_order['coc'])

    df_chat['review_or_remarks'] = df_chat['review_or_remarks'].apply(lambda x: processing.text_preprocessing(x))
    df_chat = df_chat[~df_chat['review_or_remarks'].str.endswith('?')]
    df_chat = postprocess_tickets(df_chat)

    return df_chat

def process_zendesk(df_zendesk_raw, mappings, metadata):
    cols_order, cols_rename = mappings.values()

    df_zendesk = df_zendesk_raw.copy()
    df_zendesk = processing.rename_columns_with_template(df_zendesk, cols_rename)
    df_zendesk['taxi_type'] = df_zendesk['taxi_type'].apply(processing.fix_taxi_types)
    df_zendesk = processing.validate_tickets(df_zendesk)
    df_zendesk = processing.apply_metadata_to_dataframe(df_zendesk, metadata)

    df_zendesk = processing.order_column_by_template(df_zendesk, cols_order['coc'])
    df_zendesk = postprocess_tickets(df_zendesk, remove_duplicate=['booking_code'])

    return df_zendesk

def process_drm(df_drm_raw, mappings, metadata, cooldowns):
    cols_order, cols_rename = mappings.values()

    df_drm = df_drm_raw.copy()
    df_drm = df_drm[df_drm['mex_open_rule'] == 'True']
    df_drm = df_drm.replace({'': 0, ' ': 0, 'NaN': 0, np.nan: 0})
    df_drm = processing.rename_columns_with_template(df_drm, cols_rename)
    df_drm = processing.validate_tickets(df_drm)
    df_drm = processing.apply_metadata_to_dataframe(df_drm, metadata)
    df_drm = striking.check_cooldown(df_drm, cooldowns, 'identifier', remove=True)
    df_drm = postprocess_tickets(df_drm)
    df_coc, df_docs = split_tickets(df_drm, cols_order, 'doc_det_drm')

    return df_coc, df_docs

def process_dsd_det(df_dsd_raw, mappings, metadata, cooldowns, thresholds):
    cols_order, cols_rename = mappings.values()
    
    df_dsd = df_dsd_raw.copy().iloc[:,:26]
    df_dsd = df_dsd.replace({'': 0, ' ': 0, 'NaN': 0, np.nan: 0})
    df_dsd = df_dsd.replace({'TRUE': 1, 'FALSE': 0})
    df_dsd = processing.rename_columns_with_template(df_dsd, cols_rename)
    df_dsd = processing.validate_tickets(df_dsd)

    df_dsd = df_dsd[
            (df_dsd['booking_state'] == 'COMPLETED') &
            (df_dsd['is_batching'] == 0) &
            (df_dsd['is_back_to_back_job'] == 0) &
            (df_dsd['fleet_name'].notnull()) &
            (~df_dsd['fleet_name'].str.upper().str.startswith(('GC','UNI','TPI'), na=False))
            ].copy()

    df_dsd['eta_ata_gap'] = (pd.to_datetime(df_dsd['ata_timestamp']) - pd.to_datetime(df_dsd['eta_timestamp'])).dt.total_seconds() / 60
    df_dsd['taxi_type'] = df_dsd['taxi_type'].replace('Kitchen', 'Mart')

    grouped = df_dsd.groupby(['driver_id', 'taxi_type'])
    df_dsd['avg_eta_ata_gap'] = grouped['eta_ata_gap'].transform('mean')
    df_dsd['total_rides'] = grouped['booking_code'].transform('nunique')
    df_dsd['total_delay'] = grouped['eta_ata_gap'].transform(lambda x: (x > 20).sum())
    df_dsd['delay_rate'] = df_dsd['total_delay'] / df_dsd['total_rides']
    df_dsd.sort_values(['taxi_type', 'avg_eta_ata_gap', 'eta_ata_gap'], 
                        ascending=[True, True, False], inplace=True)
    
    df_dsd = df_dsd[
        (df_dsd['total_rides'] >= thresholds['min_rides']) &
        (df_dsd['delay_rate'] > thresholds['min_rate']) &
        (df_dsd['avg_eta_ata_gap'] > thresholds['min_gap'])
    ]

    format_cols = ['eta_ata_gap', 'avg_eta_ata_gap', 'delay_rate']
    df_dsd[format_cols] = df_dsd[format_cols].round(2)
    
    df_dsd = processing.apply_metadata_to_dataframe(df_dsd, metadata)
    df_dsd = striking.check_cooldown(df_dsd, cooldowns, 'identifier', remove=True)
    df_dsd = postprocess_tickets(df_dsd, identifier='identifier')
    df_coc, df_docs = split_tickets(df_dsd, cols_order, 'doc_zen_dsd')

    return df_coc, df_docs

def process_dsd_zen(df_dsd_raw, mappings, metadata, cooldowns):
    cols_order, cols_rename = mappings.values()
    
    df_dsd = df_dsd_raw.copy()
    df_dsd = processing.rename_columns_with_template(df_dsd, cols_rename)
    df_dsd = processing.validate_tickets(df_dsd)

    df_dsd['driver_id'] = pd.to_numeric(df_dsd['driver_id'], errors='coerce').astype('Int64') 
    df_dsd['city_name'] = df_dsd['country_city'].apply(lambda x: x.split('__')[-1].capitalize())
    df_dsd['taxi_type'] = df_dsd['taxi_type'].apply(processing.fix_taxi_types)

    df_dsd = df_dsd[df_dsd['booking_code'].str.match('^(A-|IN-|SD-|MS-)')]
    df_dsd = df_dsd[df_dsd['daxdelaywoinform'] == 'True']
    df_dsd = df_dsd[df_dsd['root_cause'].str.contains('mins late')]
    df_dsd = df_dsd[df_dsd['driver_id'] > 0]

    df_dsd = processing.apply_metadata_to_dataframe(df_dsd, metadata)
    df_dsd = striking.check_cooldown(df_dsd, cooldowns, 'identifier', remove=True)
    df_dsd = postprocess_tickets(df_dsd, identifier='identifier')

    df_coc, df_docs = split_tickets(df_dsd, cols_order, 'doc_zen_dsd')

    return df_coc, df_docs

def process_dunc(df_dunc_raw, mappings, metadata, cooldowns):
    cols_order, cols_rename = mappings.values()
    
    df_dunc = df_dunc_raw.copy()
    df_dunc = df_dunc.drop(columns='vertical', axis=1)
    df_dunc = df_dunc.replace({'': 0, ' ': 0, 'NaN': 0, np.nan: 0})
    df_dunc = processing.rename_columns_with_template(df_dunc, cols_rename)
    df_dunc['taxi_type'] = df_dunc['taxi_type'].apply(processing.fix_taxi_types)
    
    df_dunc = processing.validate_tickets(df_dunc)
    df_dunc = processing.apply_metadata_to_dataframe(df_dunc, metadata)

    df_dunc = striking.check_cooldown(df_dunc, cooldowns, 'identifier', remove=True)
    df_dunc = postprocess_tickets(df_dunc, identifier='identifier')
    df_coc, df_docs = split_tickets(df_dunc, cols_order, 'doc_det_dunc')

    return df_coc, df_docs

def process_dipc(df_dipc_raw, mappings, metadata, cooldowns, source):
    cols_order, cols_rename = mappings.values()
    
    df_dipc = df_dipc_raw.copy()
    df_dipc = df_dipc.replace({'': 0, ' ': 0, 'NaN': 0, np.nan: 0})
    df_dipc = processing.rename_columns_with_template(df_dipc, cols_rename)
    df_dipc['taxi_type'] = df_dipc['taxi_type'].apply(processing.fix_taxi_types)
    
    df_dipc = processing.validate_tickets(df_dipc)
    df_dipc = processing.apply_metadata_to_dataframe(df_dipc, metadata)
    
    df_dipc.drop_duplicates('booking_code', inplace=True)
    df_dipc['repeat'] = df_dipc['identifier'].map(df_dipc['identifier'].value_counts())
    df_dipc = df_dipc[df_dipc['repeat'] > 1]

    df_dipc = striking.check_cooldown(df_dipc, cooldowns, 'identifier', remove=True)
    df_dipc = postprocess_tickets(df_dipc, identifier='identifier')
    df_coc, df_docs = split_tickets(df_dipc, cols_order, source)

    return df_coc, df_docs
