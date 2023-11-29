from datetime import datetime, timedelta

def get_last_week_date_range(start_day="monday"):
    current_date = datetime.now()

    if start_day == 'monday':
        current_start_day = current_date - timedelta(days=current_date.weekday())
    elif start_day == 'sunday':
        current_start_day = current_date - timedelta(days=current_date.weekday() + 1)

    previous_week_start = current_start_day - timedelta(weeks=1)
    previous_week_end = previous_week_start + timedelta(days=6)
    
    date_start = previous_week_start.strftime('%Y-%m-%d')
    date_end = previous_week_end.strftime('%Y-%m-%d')
    
    return date_start, date_end

def get_n_last_date_range(n):
    today_date = datetime.today()
    date_format = "%Y-%m-%d"
    
    if n == 0:
        date_start = date_end = today_date.strftime(date_format)
    elif n == 1:
        yesterday_date = today_date - datetime.timedelta(days=1)
        date_start = yesterday_date.strftime(date_format)
        date_end = today_date.strftime(date_format)
    else: 
        date_start = (today_date - timedelta(days=n)).strftime(date_format)
        date_end = today_date.strftime(date_format)

    return date_start, date_end

def get_date_range(start_date, end_date):
    date_list = []
    current_date = datetime.strptime(start_date, '%Y-%m-%d')

    while current_date <= datetime.strptime(end_date, '%Y-%m-%d'):
        date_list.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)

    return date_list

def get_last_week_monday():
    today = datetime.now().date()
    last_week_monday = today - timedelta(days=today.weekday() + 7)
    return last_week_monday.strftime('%Y-%m-%d')

def get_weekly_date_range(start_week, end_week, start_day='monday'):
    current_date = datetime.now()
    date_ranges = []

    for week_number in range(start_week, end_week + 1):
        current_start_day = datetime(current_date.year, 1, 1) + timedelta(weeks=week_number - 1)
        current_start_day -= timedelta(days=current_start_day.weekday() + (start_day == 'sunday'))

        current_end_day = current_start_day + timedelta(days=6)
        date_ranges.append((current_start_day.strftime('%Y-%m-%d'), current_end_day.strftime('%Y-%m-%d')))

    date_start, date_end = min(date_ranges, key=lambda x: x[0])[0], max(date_ranges, key=lambda x: x[1])[1]
    
    return date_start, date_end

def get_monthly_date_range(start_month, end_month):
    current_date = datetime.now()
    current_start_day = datetime(current_date.year, start_month, 1)
    last_day_of_end_month = (datetime(current_date.year, end_month % 12 + 1, 1) - timedelta(days=1)).day
    current_end_day = datetime(current_date.year, end_month, last_day_of_end_month)
    
    date_start = current_start_day.strftime('%Y-%m-%d')
    date_end = current_end_day.strftime('%Y-%m-%d')
    
    return date_start, date_end
