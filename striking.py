import numpy as np
import pandas as pd
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
