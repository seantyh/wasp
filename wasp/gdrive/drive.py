from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

def list_drive_folder(folder_id, creds):    
    service = build('drive', 'v3', credentials=creds)
    files = get_files_in_folder(service, folder_id)
    return files

def get_files_in_folder(service, folder_id):
    """Print files belonging to a folder.

    Args:
    service: Drive API service instance.
    folder_id: ID of the folder to print files from.
    """
    page_token = None
    files = []
    while True:
        try:
            param = {}
            if page_token:
                param['pageToken'] = page_token
            children = service.files().list(
                # q="name = 'model_test'",
                q=f"'{folder_id}' in parents",
                fields='nextPageToken, files(id, name)',
                includeItemsFromAllDrives=True,
                supportsAllDrives=True).execute()
            
            for child in children.get('files', []):
                # print('%s(%s)' % (child.get('id'), child.get('name')))
                files.append((child.get('id'), child.get('name')))
            break
            page_token = children.get('nextPageToken')
            if not page_token:
                break        
        except HttpError as error:
            print('An error occurred: %s' % error)
            break
    return files