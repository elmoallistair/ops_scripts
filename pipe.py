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
