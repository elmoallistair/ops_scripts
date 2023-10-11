import json
import gspread
import numpy as np
import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build

def authenticate_client(json_key_file_path):
    """
    Authenticate a client application to access Google Sheets and Google Drive using a service account.

    Args:
        json_key_file_path (str): The file path to a service account JSON key file obtained from the Google Cloud Console.

    Returns:
        gspread.client.Client or None:
            An authorized gspread client object if authentication is successful,
            or None if authentication fails.

    Example: 
        authenticate_client('your_service_account_key.json')
    """
    scope = ['https://www.googleapis.com/auth/drive', 'https://www.googleapis.com/auth/spreadsheets']
    try:
        credentials = service_account.Credentials.from_service_account_file(
            json_key_file_path, scopes=scope
        )
        print(f'Service account "{credentials.service_account_email}" successfully authenticated')
        return gspread.authorize(credentials)
    except Exception as e:
        print(f'Error authenticating client: {str(e)}')
        return None


def read_data_from_sheet(client, sheet_id, sheet_name, verbose=False):
    """
    Read data from one or more sheets within a Google Sheets document using an authenticated client.

    Args:
        client (gspread.client.Client): An authenticated gspread client.
        sheet_id (str): The ID of the Google Sheets document to read data from.
        sheet_name (str or list of str): The name(s) of the sheet(s) within the document to read data from.
        verbose (bool, optional): If True, print a success message after reading data (default is True).

    Returns:
        pd.DataFrame or None: A pandas DataFrame containing the concatenated data from the specified sheet(s) if successful,
        or None if no valid data frames were found. 

    Example:
        df = read_data_from_sheet(your_authenticated_client, 'your_spreadsheet_id', ['your_spreadsheet_name(s)'])
    """
    
    spreadsheet = client.open_by_key(sheet_id)

    if isinstance(sheet_name, str):
        sheet_name = [sheet_name]

    data_frames = []
    for name in sheet_name:
        try:
            worksheet = spreadsheet.worksheet(name)
            data = worksheet.get_all_values()
            dataframe = pd.DataFrame(data[1:], columns=data[0])
            data_frames.append(dataframe)
        except gspread.exceptions.WorksheetNotFound:
            print(f'[Warning] Sheet {name} does not exist')

    if not data_frames:
        return None

    dataframe = pd.concat(data_frames, axis=0, ignore_index=True)

    if verbose:
        print(f'Successfully read {len(dataframe)} rows from "{sheet_name}"')

    return dataframe

def write_dataframe_to_sheet(client, df_source, sheet_id, sheet_name, values_only=False, clear_sheet=True, verbose=True):
    """
    Write a Pandas DataFrame to a specified sheet within a Google Sheets document using an authenticated client.

    Args:
        client (gspread.client.Client): An authenticated gspread client for accessing Google Sheets.
        df_source (pd.DataFrame): The Pandas DataFrame containing the data to be written to the sheet.
        sheet_id (str): The unique identifier of the Google Sheets document (spreadsheet) to write data to.
        sheet_name (str): The name of the sheet within the document where data will be written.
        values_only (bool, optional): Whether to write values only (excluding column headers). Defaults to False.
        clear_sheet (bool, optional): Whether to clear the existing content of the sheet before writing data. Defaults to True.
        verbose (bool, optional): Whether to print status messages. Defaults to True.

    Example:
        write_dataframe_to_sheet(your_authenticated_client, your_dataframe, 'your_spreadsheet_id', 'your_sheet_name')
    """
    dataframe = df_source.copy()
    dataframe = dataframe.fillna('')
    dataframe = dataframe.astype(str)
    spreadsheet = client.open_by_key(sheet_id)

    try:
        worksheet = spreadsheet.worksheet(sheet_name)
    except gspread.exceptions.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(sheet_name, rows=1, cols=1)

    if clear_sheet:
        worksheet.clear()

    if values_only:
        worksheet.update('A2', dataframe.values.tolist(), value_input_option='USER_ENTERED')
    else:
        worksheet.update('A1', [dataframe.columns.tolist()] + dataframe.values.tolist(), value_input_option='USER_ENTERED')

    if verbose:
        print(f'Successfully wrote {len(dataframe)} rows to sheet "{sheet_name}"')
