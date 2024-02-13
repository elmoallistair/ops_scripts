import numpy as np
import pandas as pd
import processing
import google_connect as gconnect

def get_cooldown_data(client, sheet_id, sheet_name, identifier_col, verbose=True):
    df_cooldown = gconnect.read_data_from_sheet(client, sheet_id, sheet_name)
    act_date = df_cooldown['action_date'].max()
    cooldowns = df_cooldown[df_cooldown['action_date'] == act_date]
    cooldowns = set(cooldowns[identifier_col])

    if verbose:
        print(f"[INFO] Retrieved {len(cooldowns)} dax in cooldown from {act_date} period")

    return cooldowns

def check_cooldown(df_tickets, cooldowns, identifier_col, remove=False):
    df_tickets['cooldown'] = False
    cooldown_ids = set(cooldowns)
    df_tickets.loc[df_tickets[identifier_col].isin(cooldown_ids), 'cooldown'] = True

    if remove:
        rows_removed = df_tickets[df_tickets['cooldown'] == True]
        rows_removed = set(rows_removed[identifier_col])
        df_tickets = df_tickets[df_tickets['cooldown'] == False]
        print(f'[INFO] Removed {len(rows_removed)} dax on cooldown period')

    return df_tickets

def check_processed_tickets(df_tickets_raw, df_record, right_id, left_id, keep=False):
    df_tickets = df_tickets_raw.copy()
    processed_tickets = set(df_record[left_id].unique())
    df_tickets['processed'] = False

    for index, row in df_tickets.iterrows():
        identifier = row[right_id]
        if identifier in processed_tickets:
            df_tickets.at[index, 'processed'] = True

    if not keep:
        df_tickets = df_tickets[df_tickets['processed'] == False]
        removed_tickets = len(df_tickets_raw) - len(df_tickets)
        if removed_tickets > 0:
            print("[INFO] Removed {} processed tickets".format(removed_tickets))
        df_tickets = df_tickets.drop(columns=['processed'])

    return df_tickets

def check_duplicated_id(df_tickets_raw, identifier_col, keep=False):
    df_tickets = df_tickets_raw.copy()
    df_tickets['duplicated_id'] = df_tickets.duplicated(subset=[identifier_col], keep=False)
    
    if not keep:
        df_tickets = df_tickets[~df_tickets['duplicated_id']]
        removed_tickets = len(df_tickets_raw) - len(df_tickets[~df_tickets['duplicated_id']])
        if removed_tickets > 0:
            print("[INFO] Removed {} duplicated rows".format(removed_tickets))
        df_tickets = df_tickets.drop(columns=['duplicated_id'])

    return df_tickets

def get_strikes_and_cooldown_data(df_strike, identifier_col, strike_col, act_date_col):
    strikes = df_strike.set_index(identifier_col)[strike_col].to_dict()
    cooldowns = df_strike[df_strike[act_date_col] == df_strike[act_date_col].max()][identifier_col]

    return strikes, cooldowns

def counting_strike(df_tickets, strikes, cooldowns, identifier_col, relaxation_col, restricted_strike=None, blacklisted_strike=None, booking_level=False):
    df = df_tickets.copy()
    
    df['cooldown'] = df[identifier_col].isin(cooldowns).map({True: True, False: False})
    df[relaxation_col] = df[relaxation_col].replace({'No': False, 'Yes': True})
    df['treatment'] = ~(df['cooldown'] | df[relaxation_col])
    
    df['curr_strike'] = df_tickets[identifier_col].map(strikes)
    df['curr_strike'].fillna(0, inplace=True)
    df['curr_strike'] = df['curr_strike'].astype(int)
    
    df['add_strike'] = df.groupby(identifier_col)[identifier_col].transform('count').astype(int)

    def calculate_new_strike(row):
        if row['treatment']:
            if row['curr_strike'] < restricted_strike:
                new_strike = min(restricted_strike, row['curr_strike'] + row['add_strike'])
            else:
                new_strike = row['curr_strike'] + row['add_strike']
                if new_strike > blacklisted_strike:
                    new_strike = blacklisted_strike
        else:
            new_strike = row['curr_strike']
            
        return new_strike

    df['new_strike'] = df.apply(calculate_new_strike, axis=1)

    return df

def append_bookings(df_strike, df_strike_append):
    df_strike = df_strike.merge(df_strike_append[['Driver ID', 'Booking Code']], on='Driver ID', how='left')
    df_strike['Booking Code'].fillna('', inplace=True)
    df_strike['Total Strike'] = df_strike['Current Strike'] + df_strike['Booking Code']

    return df_strike

def check_duplicated_identifier(df_ticket, identifier='identifier_strike', reject=True):
    num_duplicates = df_ticket[identifier].duplicated().sum()
    
    if num_duplicates > 0:
        if reject:
            print(f'[REJECTED] There are {num_duplicates} duplicates in {identifier}')
            return False
        else:
            print(f'[WARNING] There are {num_duplicates} duplicates in {identifier} - Ignore IF booking-level strike')
    return True

def create_staging_for_docs(client, df_staging, sheet_engine, worksheet_target, cols_order, allowed_verticals, reject_duplicate=True):
    if not check_duplicated_identifier(df_staging, reject=reject_duplicate): 
        return None

    df_staging = processing.order_column_by_template(df_staging, cols_order)
    df_staging = df_staging[df_staging['driver_id'].replace('', np.nan).notna()]
    df_staging = df_staging[df_staging.vertical.isin(allowed_verticals)]
    df_staging = df_staging.sort_values(by=['strike', 'date_local', 'root_cause', 'sub_category'], ascending=True)
    df_staging['date_local'] = pd.to_datetime(df_staging['date_local'])
    df_staging['week'] = df_staging['date_local'].dt.isocalendar().week
    df_staging['month'] = df_staging['date_local'].dt.month
    df_staging.fillna('0', inplace=True)

    gconnect.write_dataframe_to_sheet(client, df_staging, sheet_engine, worksheet_target, 
                                      sort_by=['strike', 'root_cause', 'driver_id'])

    return df_staging

def create_staging_for_comms(client, df_staging, sheet_engine, worksheet_target, cols_order, cols_mapping, remove_duplicate=None):
    df_modified = df_staging.copy()

    mask = ~df_modified['booking_code'].str.startswith('A')
    df_modified.loc[mask, 'order_id'] = df_modified.loc[mask, 'booking_code']
    df_modified.loc[mask, 'booking_code'] = ''

    df_modified = df_modified[df_modified['final_treatment'] != 'Sharing Treatment with 2W']
    df_modified = df_modified[df_modified['treatment'] != 'Hold']

    df_comms = processing.order_column_by_template(df_modified, cols_order)
    df_comms.loc[df_comms['review_source'] == 'Comments', 'review_or_remarks'] = ''
    df_comms.sort_values(by=['strike', 'review_source', 'date_local', 'root_cause', 'sub_category', 'category'], inplace=True)
    df_comms.rename(columns=cols_mapping, inplace=True)
    df_comms.fillna('', inplace=True)

    gconnect.write_dataframe_to_sheet(client, df_comms, sheet_engine, worksheet_target, remove_duplicate=remove_duplicate, 
                                      sort_by=['Strikes', 'Root Cause', 'Driver ID'])
    return df_comms


def create_staging_for_cooldown(client, df_staging, sheet_engine, worksheet_target, identifier='identifier_strike'):
    df_cooldown = processing.order_column_by_template(df_staging, [identifier, 'frequency', 'action_date'])
        
    gconnect.write_dataframe_to_sheet(client, df_cooldown, sheet_engine, worksheet_target)

    return df_cooldown

def staging_for_sheet_strikes(client, df_staging, sheet_engine, worksheet_target, cols_order):
    df_staging = processing.order_column_by_template(df_staging, cols_order)
    df_staging = df_staging.sort_values(by=['date_local', 'vertical', 'sub_category'])
    df_staging.rename(columns={'identifier_strike': 'identifier'}, inplace=True)

    gconnect.write_dataframe_to_sheet(client, df_staging, sheet_engine, worksheet_target)

    return df_staging

def create_staging_for_remarks(client, df_staging, sheet_engine, worksheet_target, cols_order):
    df_staging = df_staging[df_staging['final_treatment'].str.contains('Blacklist|Restrict')]
    df_staging = df_staging.drop_duplicates(subset=['identifier_strike'])
    df_staging = processing.order_column_by_template(df_staging, cols_order)

    gconnect.write_dataframe_to_sheet(client, df_staging, sheet_engine, worksheet_target, 
                                      sort_by=['final_treatment', 'driver_id'])

    return df_staging

def batch_check_latest_action_dates(client, sheet_configs):
    # Iterate over each sheet configuration
    for sheet_name, (sheet_id, worksheet_name) in sheet_configs.items():
        # Read the data from the sheet
        df_content = gconnect.read_data_from_sheet(client, sheet_id, worksheet_name, verbose=False)

        # Determine the name of the action date column
        action_date_column = 'action_date' if 'action_date' in df_content.columns else 'Action Date'

        # Calculate the latest action date and the number of rows with that date
        max_action_date = df_content[action_date_column].max()
        num_of_rows = df_content[action_date_column].eq(max_action_date).sum()
        print(f'[INFO] {sheet_name} - found {num_of_rows} rows with action date of {max_action_date}')

def striking_recap(client, df_remarks, config):
    # Replace empty values with NaN
    df_remarks.replace(['', ' ', 'NA'], np.nan, inplace=True)
    df_remarks.fillna(0, inplace=True)

    # Check if all action_date are set to today's date
    today = datetime.now().strftime('%Y-%m-%d')
    df_remarks['action_date'] = pd.to_datetime(df_remarks['action_date'])
    if not all(df_remarks['action_date'] == today):
        print("[REJECTED] all action_date must be set to today's date")
        return None
    
    # Convert columns to integers
    df_remarks['strike'] = df_remarks['strike'].astype(int)
    df_remarks['fleet_id'] = df_remarks['fleet_id'].astype(int)
    df_remarks['reason_id'] = df_remarks['reason_id'].astype(int)
    df_remarks['final_course_id'] = df_remarks['final_course_id'].astype(int)

    # Input Validation
    has_empty_reason_id = (df_remarks['reason_id'] == 0).any()
    has_empty_fleet = (df_remarks['fleet_id'] == 0).any()
    has_less_than_2_strikes = (df_remarks['strike'] < 2).any()
    has_duplicate_driver_id = df_remarks.duplicated(subset='driver_id').any()

    if has_empty_reason_id:
        reason_id_indices = df_remarks[df_remarks['reason_id'] == 0].index + 1
        print(f"[WARNING] Empty reason_id found at indices: {', '.join(map(str, reason_id_indices))}.")
    
    if has_empty_fleet:
        fleet_indices = df_remarks[df_remarks['fleet_id'] == 0].index + 1
        print(f"[WARNING] Empty fleet found at indices: {', '.join(map(str, fleet_indices))}.")
    
    if has_less_than_2_strikes:
        strike_indices = df_remarks[df_remarks['strike'] < 2].index + 1
        print(f"[WARNING] Strike count less than 2 found at indices: {', '.join(map(str, strike_indices))}.")
    
    if has_duplicate_driver_id:
        duplicate_indices = df_remarks[df_remarks.duplicated(subset='driver_id')].index + 1
        print(f"[WARNING] Duplicated identifier found at indices: {', '.join(map(str, duplicate_indices))}.")

    if not (has_empty_reason_id or has_empty_fleet or has_less_than_2_strikes):
        df_restrict = df_remarks[df_remarks['final_treatment'].str.contains('Restrict')]
        df_blacklist = df_remarks[df_remarks['final_treatment'].str.contains('Blacklist')]

        # Manual Restrict
        df_restricted_manual = df_restrict[df_restrict['final_course_id'] == 0]
        df_restricted_manual['append_driver_profile_remark'] = df_restricted_manual['remarks']
        if len(df_restricted_manual) > 0:
            gconnect.append_dataframe_to_sheet(client, df_restricted_manual, config['restrict_manual']['recap_sheet_id'], 
                                               worksheet_name=config['restrict_manual']['recap_worksheet_name'], 
                                               remove_duplicate=['append_driver_profile_remark'], verbose=True)

        # Auto Restrict
        df_restricted_auto = df_restrict[df_restrict['final_course_id'] != 0]
        df_restricted_auto.rename(columns={'reason_id':'restrict_reason_id'}, inplace=True)
        if len(df_restricted_auto) > 0:
            gconnect.append_dataframe_to_sheet(client, df_restricted_auto, config['restrict_auto']['recap_sheet_id'], 
                                               worksheet_name=config['restrict_auto']['recap_worksheet_name'], 
                                               remove_duplicate=['remarks'], verbose=True)
            gconnect.append_dataframe_to_sheet(client, df_restricted_auto, config['restrict_auto']['unicorn_sheet_id'], 
                                               worksheet_name=config['restrict_auto']['unicorn_worksheet_name'], 
                                               remove_duplicate=['remarks'], verbose=True)
        # Blacklist - Recap
        df_blacklist_recap = df_blacklist.copy()
        df_blacklist_recap['date_local'] = df_blacklist_recap['action_date']
        if len(df_blacklist_recap) > 0:
            gconnect.append_dataframe_to_sheet(client, df_blacklist_recap, config['blacklist']['recap_ext_sheet_id'], 
                                               worksheet_name=config['blacklist']['recap_ext_worksheet_name'], 
                                               remove_duplicate=['driver_id', 'sub_category', 'action_date'], verbose=True)

        # Blacklist - Unicorn
        df_blacklisted = df_blacklist.copy()
        df_blacklisted['append_driver_profile_remark'] = df_blacklisted['remarks']
        if len(df_blacklisted) > 0:
            gconnect.append_dataframe_to_sheet(client, df_blacklisted, config['blacklist']['recap_sheet_id'], 
                                               worksheet_name=config['blacklist']['recap_worksheet_name'], 
                                               remove_duplicate=['append_driver_profile_remark'], verbose=True)

        # Blacklist - Update Taxi Type
        df_update_taxi = df_blacklist.copy()
        df_update_taxi['activated'] = 0
        if len(df_update_taxi) > 0:
            gconnect.append_dataframe_to_sheet(client, df_update_taxi, config['update_taxi_type']['recap_sheet_id'], 
                                               worksheet_name=config['update_taxi_type']['recap_worksheet_name'], 
                                               remove_duplicate=['driver_id', 'fleet_id', 'action_date'], verbose=True)

        return df_restricted_auto, df_restricted_manual, df_blacklisted, df_update_taxi, df_blacklist_recap

    print ('[REJECTED] Please fix the errors above before proceeding.')
    return None
