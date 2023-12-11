import os
import gspread
import pandas as pd
from datetime import datetime
from typing import Union, List, Optional
from google.oauth2 import service_account
from googleapiclient.discovery import build

def authenticate_client(json_key_file_path: str) -> gspread.Client:
    """
    Authenticates a client application to access Google Sheets and Google Drive using a service account.

    Args:
        json_key_file_path (str): The file path to a service account JSON key file obtained from the Google Cloud Console.

    Returns:
        An authorized gspread client object
    """

    # Define the required scopes for accessing Google Drive and Google Sheets
    scope = [
        'https://www.googleapis.com/auth/drive',
        'https://www.googleapis.com/auth/spreadsheets'
    ]

    # Create service account credentials using the provided JSON key file
    credentials = service_account.Credentials.from_service_account_file(
        json_key_file_path, scopes=scope
    )

    # Authorize a gspread client using the obtained credentials
    client = gspread.authorize(credentials)
    service_account_email = credentials.service_account_email
    print(f"[SUCCESS] Service account '{service_account_email}' authenticated")

    return client

def read_data_from_sheet(client: gspread.client.Client, sheet_id: str, sheet_name: Union[str, List[str]], verbose: bool = False) -> Optional[pd.DataFrame]:
    """
    Read data from one or more sheets within a Google Sheets document.

    Args:
        client (gspread.client.Client): An authenticated gspread client.
        sheet_id (str): The ID of the Google Sheets document to read data from.
        sheet_name (str or list of str): The name(s) of the sheet(s) within the document to read data from.
        verbose (bool, optional): If True, print a success message after reading data (default is False).

    Returns:
        pd.DataFrame or None: A pandas DataFrame containing the concatenated data from the specified sheet(s) if successful,
        or None if no valid data frames were found.
    """

    # Open the Google Sheets document using the provided sheet_id
    spreadsheet = client.open_by_key(sheet_id)

    if isinstance(sheet_name, str):
        sheet_name = [sheet_name]

    dataframes = []
    for name in sheet_name:
        try:
            worksheet = spreadsheet.worksheet(name) # Get the worksheet by name
            data = worksheet.get_all_values() # Get all values from the worksheet
            dataframe = pd.DataFrame(data[1:], columns=data[0]) # Create a DataFrame from the data, excluding the header row
            dataframes.append(dataframe)
        except gspread.exceptions.WorksheetNotFound:
            print(f"[Warning] Sheet '{name}' does not exist")

    if not dataframes:
        return None

    # Concatenate the dataframes vertically
    dataframe = pd.concat(dataframes, axis=0, ignore_index=True)

    if verbose:
        sheet_names = sheet_name if isinstance(sheet_name, str) else ', '.join(sheet_name)
        print(f'[INFO] Retrieved {len(dataframe)} rows from sheet "{sheet_names}"')

    return dataframe

def write_dataframe_to_sheet(client: gspread.client.Client, df_source: pd.DataFrame, sheet_id: str, sheet_name: str, values_only: bool = False, clear_sheet: bool = True, verbose: bool = True) -> None:
    """
    Write a Pandas DataFrame to a specified sheet within a Google Sheets document.

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
    # Make a copy of the source DataFrame to avoid modifying the original data
    dataframe = df_source.copy()

    # Fill any missing values in the DataFrame with empty strings
    dataframe = dataframe.fillna('')

    # Convert all values in the DataFrame to strings
    dataframe = dataframe.astype(str)

    # Open the Google Sheets document using the provided sheet_id
    spreadsheet = client.open_by_key(sheet_id)

    try:
        # Get the worksheet by name
        worksheet = spreadsheet.worksheet(sheet_name)
    except gspread.exceptions.WorksheetNotFound:
        # If the worksheet doesn't exist, add a new worksheet with the specified name
        worksheet = spreadsheet.add_worksheet(sheet_name, rows=1, cols=1)

    if clear_sheet:
        # Clear the existing content of the sheet if clear_sheet is True
        worksheet.clear()

    if values_only:
        # Write only the values (excluding column headers) to the worksheet starting from cell A2
        worksheet.update('A2', dataframe.values.tolist(), value_input_option='USER_ENTERED')
    else:
        # Write both the column headers and values to the worksheet starting from cell A1
        worksheet.update('A1', [dataframe.columns.tolist()] + dataframe.values.tolist(), value_input_option='USER_ENTERED')

    if verbose:
        print(f'[INFO] Wrote {len(dataframe)} rows to sheet "{sheet_name}"')

def append_dataframe_to_sheet(client: gspread.client.Client, df_to_append: pd.DataFrame, sheet_id: str, sheet_name: str, verbose: bool = True, remove_duplicate: Optional[Union[str, List[str]]] = None, days_limit: Optional[int] = None, sort_by: Optional[List[str]] = None) -> None:
    """
    Append a Pandas DataFrame to an existing sheet within a Google Sheets document.

    Args:
        client (gspread.client.Client): An authenticated gspread client for accessing Google Sheets.
        df_to_append (pd.DataFrame): The Pandas DataFrame containing the data to be appended to the sheet.
        sheet_id (str): The unique identifier of the Google Sheets document (spreadsheet) to append data to.
        sheet_name (str): The name of the sheet within the document where data will be appended.
        verbose (bool, optional): Whether to print status messages. Defaults to True.
        remove_duplicate (str or list, optional): Column name(s) for removing duplicate rows. Defaults to None.
        days_limit (int, optional): Number of days to limit the data to. Defaults to None (no filtering).
        sort_by (list, optional): List of column names to sort the DataFrame by. Defaults to None (no sorting).
    """
    # Read the existing data from the sheet
    df_existing = read_data_from_sheet(client, sheet_id, sheet_name)

    # Concatenate the existing data with the new data to be appended
    df_combined = pd.concat([df_to_append, df_existing], ignore_index=True)

    if remove_duplicate:
        # Remove duplicate rows based on the specified column(s)
        if isinstance(remove_duplicate, str):
            remove_duplicate = [remove_duplicate]
        df_combined = df_combined.drop_duplicates(subset=remove_duplicate)

    if days_limit:
        # Filter the data based on the specified number of days
        date_start = datetime.now() - timedelta(days=days_limit)
        df_combined['date_local'] = pd.to_datetime(df_combined['date_local'])
        df_combined = df_combined[df_combined['date_local'] >= date_start]
        df_combined['date_local'] = pd.to_datetime(df_combined['date_local'], format='%Y-%m-%d').dt.date

    if sort_by:
        if isinstance(sort_by, list):
            # Sort the DataFrame based on the specified column(s)
            df_combined = df_combined.sort_values(by=sort_by, ignore_index=True)

    if verbose:
        print(f'[INFO] Appended {len(df_combined)} rows to sheet "{sheet_name}"')

    # Write the combined DataFrame to the sheet
    write_dataframe_to_sheet(client, df_combined, sheet_id, sheet_name, verbose=False)

