import striking
import processing
import numpy as np
import pandas as pd

def split_tickets(df_tickets, cols_order, identifier):
    df_tickets_coc = df_tickets.drop_duplicates('identifier')
    df_tickets_coc = processing.order_column_by_template(df_tickets_coc, cols_order['coc'])
    df_tickets_docs = processing.order_column_by_template(df_tickets, cols_order[identifier])

    return df_tickets_coc, df_tickets_docs

def process_sdzd(df_sdgs_raw, mappings, metadata, cooldowns):
    cols_order, cols_rename = mappings.values()
    
    df_sdgs = df_sdgs_raw.copy()
    df_sdgs = processing.rename_columns_with_template(df_sdgs, cols_rename)
    df_sdgs['driver_id'] = pd.to_numeric(df_sdgs['driver_id'], errors='coerce').astype('Int64') 
    df_sdgs['city_name'] = df_sdgs['country_city'].apply(lambda x: x.split('__')[-1].capitalize())
    df_sdgs['taxi_type'] = df_sdgs['taxi_type'].apply(processing.fix_taxi_types)

    df_sdgs = df_sdgs[df_sdgs['booking_code'].str.match('^(A-|IN-|SD-|MS-)')]
    df_sdgs = df_sdgs[df_sdgs['daxdelaywoinform'] == 'True']
    df_sdgs = df_sdgs[df_sdgs['root_cause'].str.contains('mins late')]
    df_sdgs = df_sdgs[df_sdgs['driver_id'] > 0]

    processing.validate_tickets(df_sdgs)
    processing.apply_metadata_to_dataframe(df_sdgs, metadata)
    df_sdgs = striking.check_cooldown(df_sdgs, cooldowns, 'identifier', remove=True)
    df_coc, df_docs = split_tickets(df_sdgs, cols_order, 'doc_zen_sd')
    print(f'Dax count : {df_coc.driver_id.nunique()}')

    return df_coc, df_docs

def process_sdau(df_sdau_raw, mappings, metadata, cooldowns, thresholds):
    cols_order, cols_rename = mappings.values()
    
    df_sdau = df_sdau_raw.copy().iloc[:,:26]
    df_sdau = df_sdau.replace({'': 0, ' ': 0, 'NaN': 0, np.nan: 0})
    df_sdau = df_sdau.replace({'TRUE': 1, 'FALSE': 0})
    df_sdau = processing.rename_columns_with_template(df_sdau, cols_rename)
    processing.validate_tickets(df_sdau)

    df_sdau = df_sdau[
            (df_sdau['booking_state'] == 'COMPLETED') &
            (df_sdau['is_batching'] == 0) &
            (df_sdau['is_back_to_back_job'] == 0) &
            (df_sdau['fleet_name'].notnull()) &
            (~df_sdau['fleet_name'].str.upper().str.startswith(('GC','UNI','TPI'), na=False))
            ].copy()

    df_sdau['eta_ata_gap'] = (pd.to_datetime(df_sdau['ata_timestamp']) - pd.to_datetime(df_sdau['eta_timestamp'])).dt.total_seconds() / 60
    df_sdau['taxi_type'] = df_sdau['taxi_type'].replace('Kitchen', 'Mart')

    grouped = df_sdau.groupby(['driver_id', 'taxi_type'])
    df_sdau['avg_eta_ata_gap'] = grouped['eta_ata_gap'].transform('mean')
    df_sdau['total_rides'] = grouped['booking_code'].transform('nunique')
    df_sdau['total_delay'] = grouped['eta_ata_gap'].transform(lambda x: (x > 20).sum())
    df_sdau['delay_rate'] = df_sdau['total_delay'] / df_sdau['total_rides']
    df_sdau.sort_values(['taxi_type', 'avg_eta_ata_gap', 'eta_ata_gap'], 
                        ascending=[True, True, False], inplace=True)
    
    df_sdau = df_sdau[
        (df_sdau['total_rides'] >= thresholds['min_rides']) &
        (df_sdau['delay_rate'] > thresholds['min_rate']) &
        (df_sdau['avg_eta_ata_gap'] > thresholds['min_gap'])
    ]

    format_cols = ['eta_ata_gap', 'avg_eta_ata_gap', 'delay_rate']
    df_sdau[format_cols] = df_sdau[format_cols].round(2)
    
    processing.apply_metadata_to_dataframe(df_sdau, metadata)
    
    df_sdau = striking.check_cooldown(df_sdau, cooldowns, 'identifier', remove=True)
    df_coc, df_docs = split_tickets(df_sdau, cols_order, 'doc_zen_sd')
    print(f'Dax count : {df_coc.driver_id.nunique()}')

    return df_coc, df_docs


def process_drm(df_drm_raw, mappings, metadata, cooldowns):
    cols_order, cols_rename = mappings.values()

    df_drm = df_drm_raw.copy()
    df_drm = df_drm[df_drm['mex_open_rule'] == 'True']
    df_drm = df_drm.replace({'': 0, ' ': 0, 'NaN': 0, np.nan: 0})
    df_drm = processing.rename_columns_with_template(df_drm, cols_rename)
    
    processing.validate_tickets(df_drm)
    processing.apply_metadata_to_dataframe(df_drm, metadata)
    
    df_drm = striking.check_cooldown(df_drm, cooldowns, 'identifier', remove=True)
    df_coc, df_docs = split_tickets(df_drm, cols_order, 'doc_zen_sd')
    print(f'Dax count : {df_coc.driver_id.nunique()}')

    return df_coc, df_docs

def process_dunc(df_dunc_raw, mappings, metadata, cooldowns):
    cols_order, cols_rename = mappings.values()
    
    df_dunc = df_dunc_raw.copy()
    df_dunc = df_dunc.drop(columns='vertical', axis=1)
    df_dunc = df_dunc.replace({'': 0, ' ': 0, 'NaN': 0, np.nan: 0})
    df_dunc = processing.rename_columns_with_template(df_dunc, cols_rename)
    df_dunc['taxi_type'] = df_dunc['taxi_type'].apply(processing.fix_taxi_types)
    
    processing.validate_tickets(df_dunc)
    processing.apply_metadata_to_dataframe(df_dunc, metadata)

    df_dunc = striking.check_cooldown(df_dunc, cooldowns, 'identifier', remove=True)
    df_coc, df_docs = split_tickets(df_dunc, cols_order, 'doc_zen_sd')
    print(f'Dax count : {df_coc.driver_id.nunique()}')

    return df_coc, df_docs

def process_dipc_gc(df_dipc_raw, mappings, metadata, cooldowns):
    cols_order, cols_rename = mappings.values()
    
    df_dipc = df_dipc_raw.copy()
    df_dipc = df_dipc.replace({'': 0, ' ': 0, 'NaN': 0, np.nan: 0})
    df_dipc = processing.rename_columns_with_template(df_dipc, cols_rename)
    df_dipc = df_dipc[df_dipc['review_or_remarks'].str.contains(r'cancel|batal', case=False, na=False)]
    df_dipc['date_local'] = df_dipc['date_local'].str[:10]
    df_dipc['taxi_type'] = df_dipc['taxi_type'].apply(processing.fix_taxi_types)
    processing.validate_tickets(df_dipc)
    processing.apply_metadata_to_dataframe(df_dipc, metadata)
    
    df_dipc.drop_duplicates('booking_code', inplace=True)
    df_dipc['repeat'] = df_dipc['identifier'].map(df_dipc['identifier'].value_counts())
    df_dipc = df_dipc[df_dipc['repeat'] > 2]

    df_dipc = striking.check_cooldown(df_dipc, cooldowns, 'identifier', remove=True)
    df_coc, df_docs = split_tickets(df_dipc, cols_order, 'doc_zen_sd')
    print(f'Dax count : {df_coc.driver_id.nunique()}')

    return df_coc, df_docs

def process_dipc_zd(df_dipc_raw, mappings, metadata, cooldowns):
    cols_order, cols_rename = mappings.values()
    
    df_dipc = df_dipc_raw.copy()
    df_dipc = df_dipc.replace({'': 0, ' ': 0, 'NaN': 0, np.nan: 0})
    df_dipc = processing.rename_columns_with_template(df_dipc, cols_rename)
    df_dipc['taxi_type'] = df_dipc['taxi_type'].apply(processing.fix_taxi_types)
    processing.validate_tickets(df_dipc)
    processing.apply_metadata_to_dataframe(df_dipc, metadata)
    
    df_dipc.drop_duplicates('booking_code', inplace=True)
    df_dipc['repeat'] = df_dipc['identifier'].map(df_dipc['identifier'].value_counts())
    df_dipc = df_dipc[df_dipc['repeat'] > 2]

    df_dipc = striking.check_cooldown(df_dipc, cooldowns, 'identifier', remove=True)
    df_coc, df_docs = split_tickets(df_dipc, cols_order, 'doc_zen_sd')
    print(f'Dax count : {df_coc.driver_id.nunique()}')

    return df_coc, df_docs


