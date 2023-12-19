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

def get_periodic_rides_data(identifier, sheet_source, period_name, period_range):
    df_rides = gconnect.read_data_from_sheet(client, sheet_source, identifier, verbose=True)
    df_rides['period'] = df_rides[period_name].astype(int)
    df_rides = df_rides[df_rides['period'].isin(period_range)]
    df_rides.drop(columns=['week', 'month'], inplace=True)

    return df_rides

def get_periodic_coc_data(df_coc, period_name, period_range, cols_order):
    df_coc['period'] = df_coc[period_name].astype(int)
    df_coc['date_local'] = pd.to_datetime(df_coc['date_local'])

    df_coc = df_coc[df_coc['remarks'] == 'Valid']
    df_coc = df_coc[df_coc['period'].isin(period_range)]

    df_coc.sort_values('date_local', inplace=True)
    df_coc.drop_duplicates(subset=['booking_code', 'source_data'], inplace=True)
    df_coc.drop_duplicates(subset=['driver_id', 'date_local', 'disposition'], inplace=True)

    print(df_coc['period'].value_counts().sort_index())
    df_coc = df_coc[cols_order]
    df_coc = df_coc[df_coc['vertical'].isin(['GrabFood', 'GrabMart', 'GrabExpress'])]
    
    return df_coc
