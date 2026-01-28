# SALIN FILE INI DAN TEST SEKARANG:

import streamlit as st
import pandas as pd
import numpy as np
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io

# CONFIG
class Config:
    FOLDER_ID = "1hX2jwUrAgi4Fr8xkcFWjCW6vbk6lsIlP"
    FILE_KSEI = "KSEI_Shareholder_Processed.csv"
    FILE_HIST = "Kompilasi_Data_1Tahun.csv"
    MAX_STOCKS = 200  # REDUCE DARI 2000!

@st.cache_data(ttl=3600)
def load_simple_data():
    """Load data sederhana tanpa kompleksitas"""
    try:
        # Init Google Drive
        creds = Credentials.from_service_account_info(
            dict(st.secrets["gcp_service_account"])
        )
        service = build('drive', 'v3', credentials=creds)
        
        # Download KSEI
        query = f"'{Config.FOLDER_ID}' in parents and name='{Config.FILE_KSEI}'"
        result = service.files().list(q=query).execute()
        file_id = result['files'][0]['id']
        
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        
        while True:
            status, done = downloader.next_chunk()
            if done:
                break
        
        fh.seek(0)
        df_ksei = pd.read_csv(fh)
        
        st.success(f"✅ Loaded KSEI: {len(df_ksei)} rows")
        return df_ksei.head(1000)  # Hanya 1000 baris untuk testing
        
    except Exception as e:
        st.error(f"Error: {e}")
        return pd.DataFrame()

def main():
    st.title("💎 HIDDEN GEM FINDER - TEST")
    
    if st.button("Load Data"):
        data = load_simple_data()
        if not data.empty:
            st.dataframe(data)
            st.success("Done!")

if __name__ == "__main__":
    main()
