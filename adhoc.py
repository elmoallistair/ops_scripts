import pandas as pd
from typing import List, Dict
from datetime import datetime, timedelta

def dax_quality_check(dax_check: List[int], dax_strike: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Perform quality check on DAX data based on driver IDs.

    Args:
        dax_check (List[int]): List of driver IDs to check.
        dax_strike (pd.DataFrame): DataFrame containing DAX strike data.
        cols (List[str]): List of columns/informations to include.

    Returns:
        pd.DataFrame: DataFrame containing the checked DAX data with the specified columns.

    """
    dax_strike['driver_id'] = pd.to_numeric(dax_strike['driver_id'], errors='coerce')
    dax_strike.dropna(subset=['driver_id'], inplace=True)

    dax_check_df = pd.DataFrame({'driver_id': dax_check})
    dax_checked = dax_strike[dax_strike.driver_id.isin(dax_check_df.driver_id)] 
    
    missing_ids = set(dax_check_df.driver_id) - set(dax_checked.driver_id)
    if missing_ids:
        print(f'[INFO] Driver ID {missing_ids} has no violation record')
    
    return dax_checked[cols]

def batch_dax_quality_check(dax_check: pd.DataFrame, dax_strike: pd.DataFrame, level_mapping: Dict[str, int], threshold: int = 2, days: int = 90) -> pd.DataFrame:
    """
    Perform batch DAX quality check for recommendation purpose.

    Args:
        dax_check (pd.DataFrame): DataFrame containing DAX to be checked.
        dax_strike (pd.DataFrame): DataFrame containing DAX strike data.
        level_mapping (Dict[str, int]): Mapping of level names to numeric values.
        threshold (int, optional): Threshold for determining recommendation. Defaults to 2.
        days (int, optional): Number of days to consider for recent data. Defaults to 90.

    Returns:
        pd.DataFrame: DataFrame with quality check results, contains recommendation verdict and violation history.
    """
    
    last_n_days = datetime.now() - timedelta(days=days)
    
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
