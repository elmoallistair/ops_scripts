import pandas as pd
from typing import Tuple
from datetime import datetime, timedelta

def get_date_range(start_date: str, end_date: str) -> list:
    """
    Generates a list of dates between start_date and end_date.

    Parameters:
    start_date (str): Start date in 'YYYY-MM-DD' format.
    end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
    list: Dates between start_date and end_date in 'YYYY-MM-DD' format.
    """
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    date_diff = (end_date - start_date).days

    date_lst = [
        (start_date + timedelta(days=i)).strftime('%Y-%m-%d') 
        for i in range(date_diff + 1)
    ]

    return date_lst

def get_last_n_period_dates(period_name: str, period_n: int) -> Tuple[str, str]:
    """
    Calculate the start and end dates for the last n periods (weeks or months) from the current date, excluding the current week or month.

    Parameters:
    period_name (str): The type of period ('week' or 'month').
    period_n (int): The number of periods to go back from the current date.

    Returns:
    Tuple[str, str]: The start and end dates of the period in 'YYYY-MM-DD' format.

    Raises:
    ValueError: If period_name is not 'week' or 'month'.
    """
    
    # Define date_end as the end of the previous week or month
    if period_name == 'week':
        date_end = (pd.Timestamp.today() - pd.DateOffset(weeks=1)) + pd.offsets.Week(weekday=6)  # Go to the next Sunday of the previous week
    elif period_name == 'month':
        date_end = (pd.Timestamp.today() - pd.DateOffset(months=1)) + pd.offsets.MonthEnd(1)  # Go to the end of the previous month
    else:
        raise ValueError("period_name must be either 'week' or 'month'")

    # Define date_start based on period_name and n
    if period_name == 'week':
        date_start = date_end - pd.DateOffset(weeks=period_n) + pd.DateOffset(days=1)  # Go to the Monday of the start week
    elif period_name == 'month':
        date_start = date_end - pd.DateOffset(months=period_n) + pd.DateOffset(days=1)  # Go to the first day of the start month
    else:
        raise ValueError("period_name must be either 'week' or 'month'")

    # Convert date_start and date_end to 'YYYY-MM-DD' format
    date_start = date_start.strftime('%Y-%m-%d')
    date_end = date_end.strftime('%Y-%m-%d')

    return date_start, date_end

def get_n_last_date_range(n: int) -> tuple:
    """
    Returns the start and end dates for the last 'n' days.

    Parameters:
    n (int): The number of days to go back from today.

    Returns:
    tuple: A tuple containing the start and end dates in 'YYYY-MM-DD' format.
    """
    today_date = datetime.today()
    date_format = "%Y-%m-%d"
    date_end = today_date.strftime(date_format)
    date_start = (today_date - timedelta(days=n)).strftime(date_format)

    return date_start, date_end

def get_last_week_date_range(start_day: str = 'monday') -> tuple:
    """
    Returns the start and end dates of the last week.
    
    Parameters:
    start_day (str): The start day of the week ('sunday' or 'monday'). Default is "monday".

    Returns:
    tuple: A tuple containing the start and end dates of the last week in 'YYYY-MM-DD' format.
    """
    current_date = datetime.now()
    days_to_subtract = current_date.weekday() + 1 if start_day == 'sunday' else current_date.weekday()

    current_start_day = current_date - timedelta(days=days_to_subtract)
    previous_week_start = current_start_day - timedelta(weeks=1)
    previous_week_end = previous_week_start + timedelta(days=6)

    return previous_week_start.strftime('%Y-%m-%d'), previous_week_end.strftime('%Y-%m-%d')

def get_last_week_first_date(start_day: str = "monday") -> str:
    """
    Returns the date of the last week's first day based on the start_day parameter.

    Parameters:
    start_day (str): The start day of the week. Default is "monday".

    Returns:
    str: Date of the last week's first day in 'YYYY-MM-DD' format.
    """
    today = datetime.today()
    days_to_subtract = today.weekday() + 7 if start_day == 'monday' else today.weekday() + 1
    last_week_first_date = (today - timedelta(days=days_to_subtract)).strftime('%Y-%m-%d')
    
    return last_week_first_date

def get_weekly_date_range(start_week: int, end_week: int, start_day: str = 'monday') -> tuple:
    """
    Returns the earliest start date and the latest end date for a range of weeks.

    Parameters:
    start_week (int): The start week number.
    end_week (int): The end week number.
    start_day (str): The start day of the week. Default is "monday".

    Returns:
    tuple: A tuple containing the earliest start date and the latest end date in 'YYYY-MM-DD' format.
    """
    current_date = datetime.now()
    date_ranges = [
        (
            (datetime(current_date.year, 1, 1) + timedelta(weeks=week_number - 1) - timedelta(days=(datetime(current_date.year, 1, 1) + timedelta(weeks=week_number - 1)).weekday() + (start_day == 'sunday'))).strftime('%Y-%m-%d'),
            (datetime(current_date.year, 1, 1) + timedelta(weeks=week_number - 1) - timedelta(days=(datetime(current_date.year, 1, 1) + timedelta(weeks=week_number - 1)).weekday() + (start_day == 'sunday')) + timedelta(days=6)).strftime('%Y-%m-%d')
        )
        for week_number in range(start_week, end_week + 1)
    ]

    date_start, date_end = min(date_ranges, key=lambda x: x[0])[0], max(date_ranges, key=lambda x: x[1])[1]
    
    return date_start, date_end

def get_monthly_date_range(start_month: int, end_month: int) -> tuple:
    """
    Returns the start date of the start_month and the end date of the end_month.

    Parameters:
    start_month (int): The start month number.
    end_month (int): The end month number.

    Returns:
    tuple: A tuple containing the start date and the end date in 'YYYY-MM-DD' format.
    """
    current_date = datetime.now()
    date_start = datetime(current_date.year, start_month, 1).strftime('%Y-%m-%d')
    date_end = (datetime(current_date.year, end_month % 12 + 1, 1) - timedelta(days=1)).strftime('%Y-%m-%d')

    return date_start, date_end
