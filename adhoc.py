import pandas as pd
from datetime import datetime, timedelta

def batch_dax_quality_check(dax_check, dax_strike, threshold=2, days=90):
    """
    Perform batch DAX quality check.

    Args:
        dax_check (pandas.DataFrame): DataFrame containing DAX to be checked.
        dax_strike (pandas.DataFrame): DataFrame containing DAX strike data.
        threshold (int, optional): Threshold for determining recommendation. Defaults to 2.
        days (int, optional): Number of days to consider for recent data. Defaults to 90.

    Returns:
        pandas.DataFrame: DataFrame with quality check results, contains recommendation verdict and violation history.
    """
    
    last_n_days = datetime.now() - timedelta(days=days)
    
    level_mapping = {'Low': 1, 'Medium': 1, 'High': 2}
    dax_strike['level_num'] = dax_strike['level'].map(level_mapping)
    dax_strike.rename(columns={'sub_category': 'disposition'}, inplace=True)
    dax_strike['date_local'] = pd.to_datetime(dax_strike['date_local'])
    dax_strike = dax_strike[dax_strike['date_local'] >= last_n_days]
    
    scores = dax_strike.groupby('driver_id')['level_num'].sum().reset_index()
    scores.rename(columns={'level_num': 'score'}, inplace=True)
    
    dax_check['driver_id'] = pd.to_numeric(dax_check['driver_id'], errors='coerce')
    dax_strike['driver_id'] = pd.to_numeric(dax_strike['driver_id'], errors='coerce')
    dax_check.dropna(subset=['driver_id'], inplace=True)
    dax_strike.dropna(subset=['driver_id'], inplace=True)
    
    dax_check['score'] = dax_check['driver_id'].map(scores.set_index('driver_id')['score']).fillna(0)
    dax_check['verdict'] = dax_check['score'].apply(lambda x: 'Recommended' if x < threshold else 'Not Recommended')
    
    driver_info = dax_strike.groupby('driver_id').agg({'disposition': ', '.join, 'booking_code': ', '.join}).to_dict()
    
    dax_check['bookings'] = dax_check['driver_id'].map(driver_info['booking_code']).fillna('')
    dax_check['dispositions'] = dax_check['driver_id'].map(driver_info['disposition']).fillna('')
    
    return dax_check
