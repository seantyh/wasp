from googleapiclient.discovery import build

def retrieve_sheet(sheet_id, creds, sheet_name=""):    
    # Call the Sheets API
    service = build('sheets', 'v4', credentials=creds)
    #pylint: disable=no-member
    sheet = service.spreadsheets()
    range_str = sheet_name + "!A:G" if sheet_name else "A:G"
    result = sheet.values().get(spreadsheetId=sheet_id,
                                range=range_str).execute()
    values = result.get('values', [])

    if not values:
        print('No data found.')
    return values

def get_sheet_name(sheet_id, creds):
    service = build('sheets', 'v4', credentials=creds)
    #pylint: disable=no-member
    sheet_meta = service.spreadsheets().get(spreadsheetId=sheet_id).execute()
    sheets = sheet_meta.get('sheets', '')
    titles = [x.get('properties', {}).get("title", "") for x in sheets]
    return titles
        
            