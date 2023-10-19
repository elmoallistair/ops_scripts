import pandas as pd
from pyhive import presto

def execute_presto_query(query, username, password):
    try:
        cursor = presto.connect(host='porta.data-engineering.myteksi.net',
                                username=f'{username};cloud=aws&mode=adhoc', 
                                password=password,
                                catalog='hive', schema='public', port=443, protocol='https').cursor()
        cursor.execute(query)
        data =  pd.DataFrame(cursor.fetchall())
        data.columns = [i[0] for i in cursor.description]
        print(f'Successfully retrieved {len(data)} rows')

    except Exception as e:
        print(f'An error occurred: {str(e)}')
        return None

