import json
import socket
import time
from typing import Any, Dict, List, Optional

import gspread
import streamlit as st
from google.auth.exceptions import TransportError
from google.oauth2 import service_account
from oauth2client.service_account import ServiceAccountCredentials

# Define the scope for Google Sheets API
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/drive.file",
]

# Google Sheet name
SHEET_NAME = "CoffeeDiseaseData"
WORKSHEET_NAME = "Detections"  # Adding a specific worksheet name

# Number of retries for network operations
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


def get_google_sheets_credentials():
    """Get Google Sheets credentials from Streamlit secrets."""
    try:
        credentials_dict = st.secrets["gcp_service_account"]
        credentials = service_account.Credentials.from_service_account_info(
            credentials_dict, scopes=SCOPES
        )
        return credentials
    except Exception as e:
        st.error(
            f"Failed to load Google Sheets credentials. Please ensure you have set up the secrets.toml file correctly. Error: {str(e)}"
        )
        raise


def get_or_create_worksheet() -> gspread.Worksheet:
    """Get or create the worksheet with retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            credentials = get_google_sheets_credentials()
            client = gspread.authorize(credentials)

            try:
                sheet = client.open(SHEET_NAME)
            except gspread.SpreadsheetNotFound:
                sheet = client.create(SHEET_NAME)
                # Share with service account email
                sheet.share(
                    credentials.service_account_email, perm_type="user", role="writer"
                )

            try:
                worksheet = sheet.worksheet(WORKSHEET_NAME)
            except gspread.WorksheetNotFound:
                worksheet = sheet.add_worksheet(WORKSHEET_NAME, 1000, 20)
                # Set up headers
                headers = [
                    "Timestamp",
                    "Location",
                    "Latitude",
                    "Longitude",
                    "Disease",
                    "Status",
                    "Notes",
                ]
                worksheet.append_row(headers)

            return worksheet

        except (TransportError, socket.gaierror) as e:
            if attempt == MAX_RETRIES - 1:
                raise Exception(
                    f"Failed to connect to Google Sheets after {MAX_RETRIES} attempts: {str(e)}"
                )
            time.sleep(RETRY_DELAY)
            continue


def save_detection_to_database(
    timestamp, location, latitude, longitude, disease, status="Detected", notes=""
):
    """Save a disease detection to Google Sheets with retry logic."""
    worksheet = get_or_create_worksheet()

    row_data = [timestamp, location, latitude, longitude, disease, status, notes]

    for attempt in range(MAX_RETRIES):
        try:
            worksheet.append_row(row_data)
            return True
        except (TransportError, socket.gaierror) as e:
            if attempt == MAX_RETRIES - 1:
                raise Exception(
                    f"Failed to save detection after {MAX_RETRIES} attempts: {str(e)}"
                )
            time.sleep(RETRY_DELAY)

    return False


# 🔹 Fetch all locations from Google Sheets for disease tracking
def fetch_all_locations() -> List[Dict[str, Any]]:
    """Fetch all locations with retry logic and explicit headers."""
    worksheet = get_or_create_worksheet()

    expected_headers = [
        "Timestamp",
        "Location",
        "Latitude",
        "Longitude",
        "Disease",
        "Status",
        "Notes",
    ]

    for attempt in range(MAX_RETRIES):
        try:
            return worksheet.get_all_records(expected_headers=expected_headers)
        except (TransportError, socket.gaierror) as e:
            if attempt == MAX_RETRIES - 1:
                raise Exception(
                    f"Failed to fetch locations after {MAX_RETRIES} attempts: {str(e)}"
                )
            time.sleep(RETRY_DELAY)
        except gspread.exceptions.GSpreadException as e:
            # If headers are not matching, reset them
            if "header row" in str(e):
                worksheet.clear()
                worksheet.append_row(expected_headers)
                return []  # Return empty list for fresh start
            raise
