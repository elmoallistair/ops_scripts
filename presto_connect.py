import numpy as np
import pandas as pd
from pyhive import presto
import time

def execute_presto_query(query, username, password, verbose=True):
    try:
        start_time = time.time()
        cursor = presto.connect(host='porta.data-engineering.myteksi.net',
                                username=f'{username};cloud=aws&mode=adhoc', 
                                password=password,
                                catalog='hive', schema='public', port=443, protocol='https').cursor()
        cursor.execute(query)
        data = pd.DataFrame(cursor.fetchall())
        data.columns = [i[0] for i in cursor.description]
        
        if verbose:
            end_time = time.time()
            execution_time = end_time - start_time
            print(f'[SUCCESS] Retrieved {len(data)} rows from Presto (Execution Time: {execution_time:.2f} seconds)')
    
        return data

    except Exception as e:
        print(f'[ERROR] {str(e)}')
        return None

def retrieve_chat_data(base_query, df_keywords, date_start, date_end, username, password):
    true_keywords = '|'.join(df_keywords['true_keywords'].replace('', np.nan).dropna().astype(str).str.strip())
    false_keywords = '|'.join(df_keywords['false_keywords'].replace('', np.nan).dropna().astype(str).str.strip())

    query = base_query.format(date_start=date_start, date_end=date_end, true_keywords=true_keywords, false_keywords=false_keywords)
    df_chat_raw = execute_presto_query(query, username, password)

    return df_chat_raw
