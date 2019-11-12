import pickle
import os.path
from ..utils import *
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

def get_credential():
    # If modifying these scopes, delete the file token.pickle.
    SCOPES = [
            'https://www.googleapis.com/auth/drive.metadata.readonly',
            'https://www.googleapis.com/auth/spreadsheets.readonly']


    """Shows basic usage of the Drive v3 API.
    Prints the names and ids of the first 10 files the user has access to.
    """
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.

    token_path = get_resource_path("credential", 'google_api_token.pkl')

    if token_path.exists():
        with token_path.open('rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                get_resource_path("credential", 'drive_api_credentials.json'), SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with token_path.open('wb') as token:
            pickle.dump(creds, token)
    
    return creds