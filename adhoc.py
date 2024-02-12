import pandas as pd
from typing import List, Dict
from datetime import datetime, timedelta

def dax_quality_check(dax_check: List[int], df_record: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Perform quality check on DAX data based on driver IDs.

    Args:
        dax_check (List[int]): List of driver IDs to check.
        df_record (pd.DataFrame): DataFrame containing DAX strike data.
        cols (List[str]): List of columns/informations to include.

    Returns:
        pd.DataFrame: DataFrame containing the checked DAX data with the specified columns.

    """
    df_record['driver_id'] = pd.to_numeric(df_record['driver_id'], errors='coerce')
    df_record.dropna(subset=['driver_id'], inplace=True)

    dax_check_df = pd.DataFrame({'driver_id': dax_check})
    dax_checked = df_record[df_record.driver_id.isin(dax_check_df.driver_id)] 
    
    missing_ids = set(dax_check_df.driver_id) - set(dax_checked.driver_id)
    if missing_ids:
        print(f'[INFO] Driver ID {missing_ids} has no violation record')
    
    return dax_checked[cols]

def batch_dax_quality_check(dax_check: pd.DataFrame, dax_record: pd.DataFrame, level_mapping: Dict[str, int], 
                            threshold: int = 2, days: int = 90) -> pd.DataFrame:
    """
    Perform batch DAX quality check for recommendation purpose.

    Args:
        dax_check (pd.DataFrame): DataFrame containing DAX to be checked.
        dax_record (pd.DataFrame): DataFrame containing all violation records.
        level_mapping (Dict[str, int]): Mapping of level names to numeric values.
        threshold (int, optional): Threshold for determining recommendation. Defaults to 2.
        days (int, optional): Number of days to consider for recent data. Defaults to 90.

    Returns:
        pd.DataFrame: DataFrame with quality check results, contains recommendation verdict and violation history.
    """
    
    # Filter dax_record for recent data
    last_n_days = datetime.now() - timedelta(days=days)
    dax_record['date_local'] = pd.to_datetime(dax_record['date_local'])
    dax_record = dax_record[dax_record['date_local'] >= last_n_days]

    # Sum scores per driver_id
    dax_record['score'] = dax_record['level'].map(level_mapping)
    dax_scores = dax_record.groupby('driver_id')['score'].sum().reset_index()
    dax_scores['score'] = dax_scores['score'].astype(int)
    
    # Convert driver_id columns to numeric
    dax_check['driver_id'] = pd.to_numeric(dax_check['driver_id'], errors='coerce')
    dax_record['driver_id'] = pd.to_numeric(dax_record['driver_id'], errors='coerce')
    dax_scores['driver_id'] = pd.to_numeric(dax_scores['driver_id'], errors='coerce')
    
    # Map scores and determine verdict
    dax_check['score'] = dax_check['driver_id'].map(dax_scores.set_index('driver_id')['score']).fillna(0)
    dax_check['verdict'] = dax_check['score'].apply(lambda x: 'Recommended' if x < threshold else 'Not Recommended')
    
    # Append additional informations
    driver_info = dax_record.groupby('driver_id').agg({'sub_category': ', '.join, 
                                                      'booking_code': ', '.join, 
                                                      'review_or_remarks': ' | '.join}).to_dict()
    dax_check['reviews'] = dax_check['driver_id'].map(driver_info['review_or_remarks']).fillna('')    
    dax_check['dispositions'] = dax_check['driver_id'].map(driver_info['sub_category']).fillna('')
    dax_check['booking_code'] = dax_check['driver_id'].map(driver_info['booking_code']).fillna('')

    return dax_check
