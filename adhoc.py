def dax_quality_check(dax_check, dax_strike, threshold=2):
    # Convert level to number
    level_mapping = {'Low': 1, 'Medium': 1, 'High': 2}
    dax_strike['level_num'] = dax_strike['level'].map(level_mapping)
    
    # Calculate the score for each driver
    scores = dax_strike.groupby('driver_id')['level_num'].sum().reset_index()
    scores.rename(columns={'level_num': 'score'}, inplace=True)
    
    # Map the scores to the 'driver_id' column in dax_check
    dax_check['score'] = dax_check['driver_id'].map(scores.set_index('driver_id')['score']).fillna(0)
    
    # Mark 'verdict' column based on score and threshold
    dax_check['verdict'] = dax_check['score'].apply(
        lambda x: 'Recommended' if x < threshold else 'Not Recommended'
    )
    
    # Create a dictionary mapping driver_id to disposition history and bookings
    driver_info = dax_strike.groupby('driver_id').agg({'disposition': ', '.join, 'booking_code': ', '.join}).to_dict()
    
    # Map the driver_id to disposition history and bookings in dax_check
    dax_check['bookings'] = dax_check['driver_id'].map(driver_info['booking_code']).fillna('')
    dax_check['dispositions'] = dax_check['driver_id'].map(driver_info['disposition']).fillna('')
    
    return dax_check
