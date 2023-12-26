import striking
import predictor
import processing
import custom_logic
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def split_tickets(df_tickets, cols_order_coc, cols_order_docs):
    df_tickets_coc = df_tickets.drop_duplicates('identifier')
    df_tickets_coc = processing.order_column_by_template(df_tickets_coc, cols_order_coc)
    df_tickets_docs = processing.order_column_by_template(df_tickets, cols_order_docs)

    return df_tickets_coc, df_tickets_docs

def preprocess_tickets(df_tickets, config):
    df_tickets = processing.rename_columns_with_template(df_tickets, config['cols_rename'])
    df_tickets = processing.validate_tickets(df_tickets, drop=True)
    df_tickets = processing.fix_taxi_type(df_tickets)
    df_tickets = processing.fix_city_name(df_tickets)
    
    df_tickets['review_source'] = config['source']
    df_tickets = df_tickets.drop_duplicates(subset='booking_code')
    df_tickets = df_tickets.reset_index(drop=True)
    
    return df_tickets

def process_pax_rating(df_prt, config):
    df_prt = preprocess_tickets(df_prt, config)
    df_prt = processing.text_preprocessing(df_prt, keep_punctuations=False)

    pred, conf = predictor.get_prediction_with_model(df_prt, config['model'], config['feature'])
    df_prt['prediction'] = pred
    df_prt['conf_score'] = conf

    df_prt = custom_logic.custom_prt_prediction_validator(df_prt)
    df_prt = processing.create_booking_metadata(df_prt)
    df_prt = processing.create_identifier(df_prt, ['passenger_id', 'review_or_remarks'])
    df_prt = processing.detect_suspicious_pax(df_prt)
    df_prt = processing.remove_short_reviews(df_prt)
    df_prt = processing.lookup_by_reference(df_prt, config['pred_rename'], 'sub_disposition', 'prediction')
    df_prt = processing.order_column_by_template(df_prt, config['cols_order'])

    return df_prt

def process_chat(df_chat, config):
    df_chat = preprocess_tickets(df_chat, config)
    df_chat = processing.text_preprocessing(df_chat, keep_punctuations=False)
    df_chat = processing.remove_short_reviews(df_chat, min_word=config['min_word'], min_letter=config['min_letter'])
    df_chat = processing.order_column_by_template(df_chat, config['cols_order'])
    df_chat = predictor.check_repetitive_dax(df_chat, identifier=['driver_id', 'review_or_remarks'])
    
    return df_chat

def process_zendesk(df_zendesk, config):
    df_zendesk = preprocess_tickets(df_zendesk, config)
    df_zendesk = processing.append_zendesk_ticket_id(df_zendesk)
    df_zendesk = processing.order_column_by_template(df_zendesk, config['cols_order'])
    
    return df_zendesk

def process_drm(df_drm, config):
    df_drm = df_drm[df_drm['mex_open_rule'] == 'True']
    df_drm = processing.rename_columns_with_template(df_drm, config['cols_rename'])
    df_drm = processing.validate_tickets(df_drm, drop=True)
    df_drm = processing.apply_metadata_to_dataframe(df_drm, config['metadata'])
    df_drm = striking.check_cooldown(df_drm, config['cooldown'])
    df_coc, df_docs = split_tickets(df_drm, config['coc_order'], config['docs_order'])

    return df_coc, df_docs

def process_dunc(df_dunc, config):
    df_dunc = processing.rename_columns_with_template(df_dunc, config['cols_rename'])
    df_dunc = processing.fix_taxi_type(df_dunc)
    df_dunc = processing.validate_tickets(df_dunc, drop=True)
    df_dunc = processing.apply_metadata_to_dataframe(df_dunc, config['metadata'])
    df_dunc = striking.check_cooldown(df_dunc, config['cooldown'])
    df_coc, df_docs = split_tickets(df_dunc, config['coc_order'], config['docs_order'])

    return df_coc, df_docs

def process_dsd_det(df_dsd, config):
    df_dsd = df_dsd.copy().iloc[:,:26].replace({'NaN':1, 'TRUE': 1, 'FALSE': 0})
    df_dsd = processing.rename_columns_with_template(df_dsd, config['cols_rename'])
    df_dsd = processing.validate_tickets(df_dsd, drop=True)

    # Filter rows based on conditions
    df_dsd = df_dsd[
        (df_dsd['booking_state'] == 'COMPLETED') &
        (df_dsd['is_batching'] != 0) &
        (df_dsd['is_back_to_back_job'] == 0) &
        (df_dsd['fleet_name'].notnull()) &
        (~df_dsd['fleet_name'].str.upper().str.startswith(('GC','UNI','TPI'), na=False))
    ].copy()

    # Calculate gap and fix taxi type
    df_dsd['eta_ata_gap'] = (pd.to_datetime(df_dsd['ata_timestamp']) - pd.to_datetime(df_dsd['eta_timestamp'])).dt.total_seconds() / 60
    df_dsd['taxi_type'] = df_dsd['taxi_type'].replace('Kitchen', 'Mart')

    # Group by 'driver_id' and 'taxi_type' and calculate new columns
    grouped = df_dsd.groupby(['driver_id', 'taxi_type'])
    df_dsd['avg_eta_ata_gap'] = grouped['eta_ata_gap'].transform('mean')
    df_dsd['total_rides'] = grouped['booking_code'].transform('nunique')
    df_dsd['total_delay'] = grouped['eta_ata_gap'].transform(lambda x: (x > 20).sum())
    df_dsd['delay_rate'] = df_dsd['total_delay'] / df_dsd['total_rides']

    # Sort values
    df_dsd.sort_values(['taxi_type', 'avg_eta_ata_gap', 'eta_ata_gap'], ascending=[True, True, False], inplace=True)

    # Filter rows based on conditions
    df_dsd = df_dsd[
        (df_dsd['total_rides'] >= config['min_rides']) &
        (df_dsd['delay_rate'] > config['min_rate']) &
        (df_dsd['avg_eta_ata_gap'] > config['min_gap'])
    ]

    # Round values
    format_cols = ['eta_ata_gap', 'avg_eta_ata_gap', 'delay_rate']
    df_dsd[format_cols] = df_dsd[format_cols].round(2)

    # Apply metadata and check cooldown
    df_dsd = processing.apply_metadata_to_dataframe(df_dsd, config['metadata'])
    df_dsd = striking.check_cooldown(df_dsd, config['cooldown'])

    # Split tickets
    df_coc, df_docs = split_tickets(df_dsd, config['coc_order'], config['docs_order'])

    return df_coc, df_docs

def process_dsd_zen(df_dsd, config):
    df_dsd = processing.rename_columns_with_template(df_dsd, config['cols_rename'])
    df_dsd = processing.validate_tickets(df_dsd, drop=True)
    df_dsd = processing.fix_taxi_type(df_dsd)
    df_dsd = processing.apply_metadata_to_dataframe(df_dsd, config['metadata'])
    df_dsd = striking.check_cooldown(df_dsd, config['cooldown'])

    df_dsd = df_dsd[df_dsd['daxdelaywoinform'] == 'True']
    df_dsd = df_dsd[df_dsd['root_cause'].str.contains('mins late')]
    df_coc, df_docs = split_tickets(df_dsd, config['coc_order'], config['docs_order'])

    return df_coc, df_docs

def process_dipc(df_dipc, config):
    def apply_threshold(df):
        df.drop_duplicates('booking_code', inplace=True)
        df['repeat'] = df['identifier'].map(df['identifier'].value_counts())
        return df[df['repeat'] >= config['min_repeat']]
    
    df_dipc = processing.rename_columns_with_template(df_dipc, config['cols_rename'])
    df_dipc = processing.fix_taxi_type(df_dipc)
    df_dipc = processing.validate_tickets(df_dipc, drop=True)
    df_dipc = processing.apply_metadata_to_dataframe(df_dipc, config['metadata'])
    df_dipc = apply_threshold(df_dipc)
    df_dipc = striking.check_cooldown(df_dipc, config['cooldown'])
    df_coc, df_docs = split_tickets(df_dipc, config['coc_order'], config['docs_order'])
    
    return df_coc, df_docs
