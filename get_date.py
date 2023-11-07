from datetime import datetime, timedelta

def get_last_week_date_range(start_day="monday"):
    """
    Retrieve previous week's date range.

    Args:
    - start_day (str, optional): The starting day of the week (monday or sunday). Default is "monday".

    Returns:
    A tuple of two strings representing the date range. The first string is the start date, and the second string is the end date.
    """
    current_date = datetime.now()
    current_weeknum = current_date.isocalendar()[1]

    if start_day == 'monday':
        current_start_day = current_date - timedelta(days=current_date.weekday())
    elif start_day == 'sunday':
        current_start_day = current_date - timedelta(days=current_date.weekday() + 1)

    previous_week_start = current_start_day - timedelta(weeks=1)
    previous_week_end = previous_week_start + timedelta(days=6)
    
    date_start = previous_week_start.strftime('%Y-%m-%d')
    date_end = previous_week_end.strftime('%Y-%m-%d')
    
    return date_start, date_end
