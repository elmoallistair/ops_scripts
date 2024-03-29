import os
import time
import gspread
import numpy as np
import pandas as pd
import processing
from typing import Union, List, Optional
from datetime import datetime, timedelta
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
    print(f"[INFO] Service account '{service_account_email}' successfully authenticated")

    return client

def read_data_from_sheet(
    client: gspread.client.Client, 
    sheet_id: str, 
    worksheet_name: Union[str, List[str]], 
    verbose: bool = False
) -> Optional[pd.DataFrame]:
    """
    Read data from one or more sheets within a Google Sheets document.

    Args:
        client (gspread.client.Client): An authenticated gspread client.
        sheet_id (str): The ID of the Google Sheets document to read data from.
        worksheet_name (str or list of str): The name(s) of the worksheet(s) within the document to read data from.
        verbose (bool, optional): If True, print a success message after reading data. Defaults to False.

    Returns:
        pd.DataFrame or None: A pandas DataFrame containing the concatenated data from the specified worksheet(s) if successful,
        or None if no valid data frames were found.
    """
    spreadsheet = client.open_by_key(sheet_id)
    worksheet_name = [worksheet_name] if isinstance(worksheet_name, str) else worksheet_name

    dataframes = []
    for name in worksheet_name:
        try:
            worksheet = spreadsheet.worksheet(name)
            data = worksheet.get_all_values()
            if not data:
                print(f"[WARNING] Worksheet '{name}' is empty")
                continue
            if len(data) < 1:
                print(f"[WARNING] Worksheet '{name}' has no headers")
                continue
            dataframe = pd.DataFrame(data[1:], columns=data[0])
            dataframes.append(dataframe)
        except gspread.exceptions.WorksheetNotFound:
            print(f"[WARNING] Worksheet '{name}' does not exist")

    if not dataframes:
        return None

    dataframe = pd.concat(dataframes, axis=0, ignore_index=True)

    if verbose:
        worksheet_names = ', '.join(worksheet_name)
        print(f'[INFO] Retrieved {len(dataframe)} rows from sheet "{worksheet_names}"')

    return dataframe

def write_dataframe_to_sheet(
    client: gspread.client.Client, 
    df_source: pd.DataFrame, 
    sheet_id: str, 
    worksheet_name: str, 
    cols_order: Optional[List[str]] = None,
    remove_duplicate: Optional[Union[str, List[str]]] = None, 
    sort_by: Optional[Union[str, List[str]]] = None, 
    sort_ascending: Union[bool, List[bool]] = True, 
    fillna: str = '', 
    verbose: bool = False,
    max_retries: int = 3
) -> None:
    """
    Write a Pandas DataFrame to a Google Sheets document.

    Args:
        client (gspread.client.Client): The gspread client.
        df_source (pd.DataFrame): The DataFrame to write.
        sheet_id (str): The ID of the sheet to write to.
        worksheet_name (str): The name of the worksheet to write to.
        cols_order (Optional[List[str]]): The order of the columns. If None, the order of the DataFrame is used.
        remove_duplicate (Optional[Union[str, List[str]]]): The columns to consider when removing duplicates. If None, all columns are used.
        sort_by (Optional[Union[str, List[str]]]): The columns to sort by. If None, the DataFrame is not sorted.
        sort_ascending (Union[bool, List[bool]]): Whether or not the sorting should be done in sort_ascending order.
        fillna (str): The value to replace NaN values with.
        verbose (bool): Whether to print verbose messages.
        max_retries (int): Maximum number of retries in case of failure.
    """

    # Copy the DataFrame, replace empty strings and single-space strings with NaN, 
    # fill NaN values with the specified value and convert all values to strings.
    df = df_source.copy().replace(['', ' '], np.nan).fillna(fillna).astype(str)
 
    # If cols_order is provided and not empty, reorder the columns
    if cols_order is not None:
        df = processing.order_column_by_template(df, cols_order)
 
    # If remove_duplicate is provided, remove duplicates
    if remove_duplicate:
        df.drop_duplicates(subset=remove_duplicate, inplace=True)
 
    # If sort_by is provided, sort the DataFrame
    if sort_by:
        df.sort_values(by=sort_by, ascending=sort_ascending, inplace=True)

    # Open the spreadsheet
    spreadsheet = client.open_by_key(sheet_id)

    # Try to get the worksheet, if it doesn't exist, create it
    try:
        worksheet = spreadsheet.worksheet(worksheet_name)
    except gspread.exceptions.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(worksheet_name, rows=1, cols=1)
    
    # Convert the DataFrame to a list of lists
    data = df.values.tolist()
    data = [df.columns.tolist()] + data

    retry_count = 0
    while retry_count < max_retries:
        try:
            # Update the worksheet with the data
            worksheet.clear()
            worksheet.update('A1', data, value_input_option='USER_ENTERED')
            # If verbose is True, print a message
            if verbose:
                print(f'[INFO] Wrote {len(df)} rows to sheet "{worksheet_name}"')
            return  # Successful update, exit the loop
        except Exception as e:
            retry_count += 1
            print(f"[FAILED] Failed to write to '{worksheet_name}'. {retry_count}/{max_retries} retries attemps remaining. Error: {type(e).__name__}")
            time.sleep(10)
    
    # If still failed after max retries, create backup in local
    backup_folder = 'backup/write_sheet'
    if not os.path.exists(backup_folder):
        os.makedirs(backup_folder)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
    backup_filename = f"{worksheet_name}_{timestamp}_backup.csv"
    backup_path = os.path.join(backup_folder, backup_filename)
    df_source.to_csv(backup_path, index=False)
    print(f"[INFO] Backup created: {backup_path}")

def append_dataframe_to_sheet(
    client: gspread.client.Client, 
    df_to_append: pd.DataFrame, 
    sheet_id: str, 
    worksheet_name: str, 
    remove_duplicate: Optional[Union[str, List[str]]] = None, 
    days_limit: Optional[int] = None, 
    sort_by: Optional[List[str]] = None, 
    ascending: Union[bool, List[bool]] = True, 
    verbose: bool = True
) -> None:
    """
    Append a Pandas DataFrame to an existing sheet within a Google Sheets document.
    """
    # Read the existing data from the sheet
    df_existing = read_data_from_sheet(client, sheet_id, worksheet_name)

    # Reindex the columns of df_to_append to match df_existing, filling missing columns with NaN
    df_to_append = df_to_append.reindex(columns=df_existing.columns)

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
        # Sort the DataFrame based on the specified column(s)
        df_combined = df_combined.sort_values(by=sort_by, ascending=ascending, ignore_index=True)

    write_dataframe_to_sheet(client, df_combined, sheet_id, worksheet_name, verbose=False)

    if verbose:
        print(f'[INFO] Appended {len(df_to_append)} rows to sheet "{worksheet_name}"')


def copy_file_to_drive(client: gspread.Client, from_source: str, file_source: str, folder_id_dest: str, 
                       filename: str, if_exist: str = 'skip') -> str:
    """
    Copy a file from a local or Google Drive file to another Google Drive folder.

    Args:
        client (gspread.client.Client): An authenticated gspread client.
        from_source (str): The source of the file to be copied. 
            - 'local': Copying local file to drive folder.
            - 'drive': Copying drive file to drive folder.
        file_source (str): 
            - The path of local file if source is from 'local'.
            - The file ID if source is from 'drive'.
        folder_id_dest (str): The ID of the destination drive folder.
        filename (str): The desired filename for the copied file in the destination folder.
        if_exist (str, optional): The action to take if the file already exists in the destination folder.
            - 'override': Replace the existing file with the copied file.
            - 'skip': Skip the copying process and do nothing. Default is 'skip'.
            - 'duplicate': Create a duplicate file with a unique name.

    Returns:
        str: The ID of the copied file.
    """
        
    drive_service = build('drive', 'v3', credentials=client.auth)

    # Get the destination folder details
    folder_metadata = drive_service.files().get(fileId=folder_id_dest, fields='name').execute()
    folder_name = folder_metadata['name']

    # Prepare new file metadata
    new_file_metadata = {
        'name': filename,
        'parents': [folder_id_dest],
    }

    # Check if a file with the same name already exists in the destination folder
    existing_files = (
        drive_service.files()
        .list(q=f'name="{filename}" and "{folder_id_dest}" in parents')
        .execute()
    )

    if existing_files.get('files'):
        existing_file_id = existing_files['files'][0]['id']

        if if_exist == 'override':
            # Replace the existing file
            drive_service.files().delete(fileId=existing_file_id).execute()
            print(f'[WARNING] File "{filename}" already exists and is being replaced.')
        elif if_exist == 'skip':
            # Don't copy the file and skip the process
            print(f'[WARNING] File "{filename}" already exists, process skipped.')
            return None
        elif if_exist == 'duplicate':
            # Create a duplicate file with a unique name
            base_name, extension = os.path.splitext(filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            duplicate_name = f"{base_name}_{timestamp}{extension}"
            new_file_metadata['name'] = duplicate_name
            print(f'[WARNING] File "{filename}" already exists, creating a duplicate file: "{duplicate_name}"')

    # Copy the file based on its source
    if from_source == 'drive':
        copied_file = drive_service.files().copy(fileId=file_source, body=new_file_metadata).execute()
        filename_origin = drive_service.files().get(fileId=copied_file['id'], fields='name').execute()['name']
        print(f'[INFO] File "{filename_origin}" copied with filename "{filename}"')
    elif from_source == 'local':
        # Verify that the local file exists
        if not os.path.exists(file_source):
            raise FileNotFoundError(f'[REJECTED] File "{file_source}" not found.')

        # Upload the file to Google Drive
        copied_file = (
            drive_service.files()
            .create(body=new_file_metadata, media_body=file_source, fields='id')
            .execute()
        )
        print(f'[INFO] File "{filename}" successfully copied to folder: "{folder_name}"')

    return copied_file['id']
