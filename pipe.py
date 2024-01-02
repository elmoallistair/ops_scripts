import processing

def get_shared_treatment_tickets(df_tickets, disposition, cols_order, fw_exclude=False):
    df_tickets = df_tickets[df_tickets.disposition == disposition]
    df_tickets.rename(columns={'sub_disposition': 'root_cause', 'disposition': 'sub_category'}, inplace=True)
    df_tickets = processing.order_column_by_template(df_tickets, cols_order)
    df_tickets.reset_index(drop=True, inplace=True)

    if fw_exclude:
        df_tickets = df_tickets[~df_tickets['fleet_name'].str.upper().str.startswith(('GC','UNI','TPI'), na=False)]

    print(f'[INFO] Retrieved {len(df_tickets)} {disposition} tickets')
    df_tickets['action_date'] = datetime.now().strftime('%Y-%m-%d')

    return df_tickets

def get_safety_and_maps_complains(df_tickets, cols_order, reporter='pax', verbose=True):
    df_maps = df_tickets[df_tickets.sub_disposition == 'Grab Maps']
    df_safety = df_tickets[df_tickets.sub_disposition == 'Safety']

    if reporter == 'pax':
        df_safety['reporter'] = 'Passenger'
        df_safety['reporter_id'] = df_safety['passenger_id']
        df_maps['reporter'] = 'Passenger'
        df_maps['reporter_id'] = df_maps['passenger_id']
    elif reporter == 'dax':
        df_safety['reporter'] = 'Driver'
        df_safety['reporter_id'] = df_safety['driver_id']
        df_maps['reporter'] = 'Passenger'
        df_maps['reporter_id'] = df_maps['passenger_id']

    df_maps['category'] = 'Grab Maps'
    df_maps['date_today'] = datetime.date.today().strftime('%Y-%m-%d')
    df_maps['action_date'] = datetime.date.today().strftime('%Y-%m-%d')
    df_safety['action_date'] = datetime.date.today().strftime('%Y-%m-%d')
    
    df_maps = processing.order_column_by_template(df_maps, cols_order['routing'])
    df_safety = processing.order_column_by_template(df_safety, cols_order['safety'])
        
    if verbose:
        print(f'[INFO] Retrieved {len(df_maps)} maps tickets')
        print(f'[INFO] Retrieved {len(df_safety)} safety tickets')

    return df_safety, df_maps

def get_periodic_rides_data(identifier, sheet_source, period_name, period_range):
    df_rides = gconnect.read_data_from_sheet(client, sheet_source, identifier, verbose=True)
    df_rides['period'] = df_rides[period_name].astype(int)
    df_rides = df_rides[df_rides['period'].isin(period_range)]
    df_rides.drop(columns=['week', 'month'], inplace=True)

    return df_rides

def get_periodic_coc_data(df_coc, period_name, period_range, cols_order):
    df_coc['period'] = df_coc[period_name].astype(int)
    df_coc['date_local'] = pd.to_datetime(df_coc['date_local'])
    df_coc = df_coc[df_coc['period'].isin(period_range)]

    df_coc.sort_values('date_local', inplace=True)
    df_coc.drop_duplicates(subset=['booking_code', 'review_source'], inplace=True)
    df_coc.drop_duplicates(subset=['driver_id', 'date_local', 'disposition'], inplace=True)

    df_coc = df_coc[cols_order]
    df_coc = df_coc[df_coc['vertical'].isin(['GrabFood', 'GrabMart', 'GrabExpress'])]
    df_coc = df_coc[df_coc['territory'].isin(['West', 'Jabo', 'East'])]
    print(df_coc['period'].value_counts().sort_index())
    
    return df_coc
