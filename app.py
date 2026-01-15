# ==============================================================================
# 📦 IMPORTS
# ==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

# Google Drive imports
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ==============================================================================
# ⚙️ PAGE CONFIG & CSS
# ==============================================================================
st.set_page_config(
    page_title="MERGE ANALYTIC KSEI & DAILY TX1Y [DEBUG]",
    layout="wide",
    page_icon="🔧",
    initial_sidebar_state="expanded"
)

# --- KSEI STYLE CSS ---
st.markdown("""
<style>
    :root {
        --primary-color: #4318FF;
        --background-color: #F4F7FE;
        --secondary-background-color: #FFFFFF;
        --text-color: #2B3674;
        --font: 'DM Sans', sans-serif;
    }
    .stApp {
        background-color: #F4F7FE;
        color: #2B3674;
    }
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        box-shadow: 14px 14px 40px rgba(112, 144, 176, 0.08);
        border-right: none;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span {
        color: #2B3674 !important;
        font-weight: 600;
    }
    div.stButton > button {
        background: linear-gradient(90deg, #4318FF 0%, #868CFF 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        width: 100%;
        box-shadow: 0px 4px 10px rgba(67, 24, 255, 0.2);
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0px 8px 15px rgba(67, 24, 255, 0.3);
        border: none;
        color: white;
    }
    .css-card {
        background-color: #FFFFFF;
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0px 18px 40px rgba(112, 144, 176, 0.12);
        margin-bottom: 24px;
        border: none;
    }
    .header-banner {
        background: linear-gradient(86.88deg, #4318FF 0%, #868CFF 100%);
        border-radius: 20px;
        padding: 30px 40px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0px 18px 40px rgba(112, 144, 176, 0.2);
    }
    .header-title { font-size: 32px; font-weight: 700; margin-bottom: 8px; }
    .header-subtitle { font-size: 16px; font-weight: 500; opacity: 0.9; }
    label {
        color: #2B3674 !important;
        font-size: 14px !important;
        font-weight: 600 !important;
    }
    .card-title {
        font-size: 20px;
        font-weight: 700;
        color: #2B3674;
        margin-bottom: 20px;
    }
    .stProgress > div > div > div > div {
        background-color: #4318FF;
    }
    .debug-box {
        background-color: #FFF5F5;
        border-left: 4px solid #EE5D50;
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 5px;
    }
    .success-box {
        background-color: #F0FFF4;
        border-left: 4px solid #05CD99;
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# ⚙️ CONFIGURATION
# ==============================================================================
FOLDER_ID = "1hX2jwUrAgi4Fr8xkcFWjCW6vbk6lsIlP"
FILE_KSEI = "KSEI_Shareholder_Processed.csv"
FILE_HIST = "Kompilasi_Data_1Tahun.csv"

# Ownership categories from KSEI data
OWNERSHIP_COLS = [
    'Local IS', 'Local CP', 'Local PF', 'Local IB', 'Local ID', 'Local MF', 'Local SC', 'Local FD', 'Local OT',
    'Foreign IS', 'Foreign CP', 'Foreign PF', 'Foreign IB', 'Foreign ID', 'Foreign MF', 'Foreign SC', 'Foreign FD', 'Foreign OT'
]
OWNERSHIP_CHG_COLS = [f"{col}_chg" for col in OWNERSHIP_COLS]
OWNERSHIP_CHG_RP_COLS = [f"{col}_chg_Rp" for col in OWNERSHIP_COLS]

# Smart money vs retail
SMART_MONEY_COLS = [
    'Foreign IS_chg_Rp', 'Foreign IB_chg_Rp', 'Foreign PF_chg_Rp', 
    'Local IS_chg_Rp', 'Local PF_chg_Rp', 'Local MF_chg_Rp', 'Local IB_chg_Rp'
]
RETAIL_COLS = ['Local ID_chg_Rp']

# Hidden Gem Scoring Weights
SCORE_WEIGHTS = {
    'smart_money': 0.35,
    'technical': 0.30,
    'fundamental': 0.25,
    'sentiment': 0.10
}

# ==============================================================================
# 📦 DATA LOADER CLASS (DEBUG VERSION)
# ==============================================================================
class DataLoader:
    def __init__(self):
        self.service = None
        self.initialize_gdrive()
    
    def initialize_gdrive(self):
        """Initialize Google Drive service with Streamlit secrets"""
        try:
            # Get credentials from secrets
            creds_data = st.secrets["gcp_service_account"]
            
            st.write("🔧 DEBUG: Checking credentials format...")
            st.write(f"Type of creds_data: {type(creds_data)}")
            
            # Handle different formats
            if isinstance(creds_data, str):
                st.write("Creds is string, parsing JSON...")
                # Clean the string if needed
                if creds_data.startswith("'") and creds_data.endswith("'"):
                    creds_data = creds_data[1:-1]
                # Replace escaped characters
                creds_data = creds_data.replace('\\n', '\n').replace('\\\\n', '\n')
                
                try:
                    creds_json = json.loads(creds_data)
                    st.success("✅ JSON parsed successfully from string")
                except json.JSONDecodeError as e:
                    st.error(f"JSON decode error: {e}")
                    # Try alternative parsing
                    try:
                        # Try with single quotes
                        import ast
                        creds_json = ast.literal_eval(creds_data)
                        st.success("✅ Parsed with ast.literal_eval")
                    except:
                        st.error("Failed all parsing attempts")
                        return False
            elif isinstance(creds_data, dict):
                st.write("Creds is already dict")
                creds_json = creds_data
            else:
                st.error(f"Unknown credentials format: {type(creds_data)}")
                return False
            
            st.write("✅ Creating credentials object...")
            creds = Credentials.from_service_account_info(
                creds_json, 
                scopes=['https://www.googleapis.com/auth/drive.readonly']
            )
            
            st.write("✅ Building GDrive service...")
            self.service = build('drive', 'v3', credentials=creds, cache_discovery=False)
            
            st.success("🎉 GDrive Service initialized successfully!")
            return True
            
        except Exception as e:
            st.error(f"❌ GDrive Auth Error: {type(e).__name__}: {str(e)}")
            import traceback
            st.code(traceback.format_exc(), language='python')
            return False
    
    def list_files_in_folder(self):
        """List all files in the folder for debugging"""
        if not self.service:
            st.warning("Service not initialized")
            return []
        
        try:
            st.write(f"🔍 Listing files in folder: {FOLDER_ID}")
            query = f"'{FOLDER_ID}' in parents and trashed=false"
            results = self.service.files().list(
                q=query, 
                fields="files(id, name, mimeType, size, modifiedTime)",
                pageSize=100
            ).execute()
            
            files = results.get('files', [])
            st.write(f"Found {len(files)} files")
            return files
            
        except Exception as e:
            st.error(f"Error listing files: {e}")
            import traceback
            st.code(traceback.format_exc())
            return []
    
    def download_file(self, file_name):
        """Download file from Google Drive with debug info"""
        if not self.service:
            return None, "Service not initialized"
        
        try:
            st.markdown(f"<div class='debug-box'><b>🔍 Searching for:</b> '{file_name}'</div>", unsafe_allow_html=True)
            
            # Search for file
            query = f"'{FOLDER_ID}' in parents and name='{file_name}' and trashed=false"
            st.write(f"Query: {query}")
            
            results = self.service.files().list(
                q=query, 
                fields="files(id, name, size, modifiedTime)",
                pageSize=5
            ).execute()
            
            items = results.get('files', [])
            
            if not items:
                st.error(f"❌ File '{file_name}' not found!")
                
                # List all available files
                st.write("📁 Available files in folder:")
                all_files = self.list_files_in_folder()
                if all_files:
                    for f in all_files:
                        st.write(f"• {f['name']} ({f.get('size', 'N/A')} bytes)")
                else:
                    st.error("No files found at all. Check folder ID and permissions.")
                
                return None, f"File '{file_name}' not found"
            
            file_info = items[0]
            st.markdown(f"<div class='success-box'><b>✅ Found file:</b> {file_info['name']}<br>ID: {file_info['id']}<br>Size: {file_info.get('size', 'N/A')} bytes</div>", unsafe_allow_html=True)
            
            # Download file
            st.write("⬇️ Starting download...")
            request = self.service.files().get_media(fileId=file_info['id'])
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    progress = int(status.progress() * 100)
                    progress_bar.progress(progress)
                    status_text.text(f"Downloading... {progress}%")
            
            progress_bar.empty()
            status_text.empty()
            
            fh.seek(0)
            file_size_mb = len(fh.getvalue()) / 1024 / 1024
            st.success(f"✅ Download complete: {file_size_mb:.2f} MB")
            return fh, None
            
        except Exception as e:
            st.error(f"❌ Download error: {type(e).__name__}: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return None, f"Download error: {e}"
    
    @st.cache_data(ttl=3600, show_spinner="🔄 Loading KSEI Data...")
    def load_ksei_data(_self):
        """Load and process KSEI ownership data"""
        st.write(f"📥 Loading KSEI file: {FILE_KSEI}")
        fh, error = _self.download_file(FILE_KSEI)
        if error:
            st.error(f"Failed to load KSEI: {error}")
            return pd.DataFrame()
        
        try:
            st.write("📊 Processing KSEI CSV...")
            df = pd.read_csv(fh, dtype=object)
            st.write(f"Raw shape: {df.shape}")
            
            # Basic cleaning
            df.columns = df.columns.str.strip()
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
            # Filter recent data
            df = df[df['Date'].dt.year >= 2024].copy()
            st.write(f"After year filter: {df.shape}")
            
            # Process numeric columns
            numeric_cols = OWNERSHIP_COLS + OWNERSHIP_CHG_COLS + OWNERSHIP_CHG_RP_COLS + ['Price', 'Free Float', 'Total_Local', 'Total_Foreign']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            
            # Calculate additional metrics
            local_cols = [c for c in OWNERSHIP_CHG_RP_COLS if 'Local' in c and c in df.columns]
            foreign_cols = [c for c in OWNERSHIP_CHG_RP_COLS if 'Foreign' in c and c in df.columns]
            
            df['Total_Local_chg_Rp'] = df[local_cols].sum(axis=1) if local_cols else 0
            df['Total_Foreign_chg_Rp'] = df[foreign_cols].sum(axis=1) if foreign_cols else 0
            df['Total_chg_Rp'] = df['Total_Local_chg_Rp'] + df['Total_Foreign_chg_Rp']
            
            # Add Smart Money total
            smart_cols = [c for c in SMART_MONEY_COLS if c in df.columns]
            df['Smart_Money_Flow'] = df[smart_cols].sum(axis=1) if smart_cols else 0
            
            # Add Retail total
            retail_cols = [c for c in RETAIL_COLS if c in df.columns]
            df['Retail_Flow'] = df[retail_cols].sum(axis=1) if retail_cols else 0
            
            # Calculate net institutional flow
            df['Institutional_Net'] = df['Smart_Money_Flow'] - df['Retail_Flow']
            
            # Ensure Stock Code column
            df['Stock Code'] = df['Code']
            
            st.success(f"✅ KSEI loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            st.write("Sample data:")
            st.dataframe(df[['Date', 'Stock Code', 'Price', 'Total_chg_Rp', 'Smart_Money_Flow']].head())
            
            return df
            
        except Exception as e:
            st.error(f"Error processing KSEI data: {e}")
            import traceback
            st.code(traceback.format_exc())
            return pd.DataFrame()
    
    @st.cache_data(ttl=3600, show_spinner="🔄 Loading Historical Data...")
    def load_historical_data(_self):
        """Load and process 1-year historical data"""
        st.write(f"📥 Loading historical file: {FILE_HIST}")
        fh, error = _self.download_file(FILE_HIST)
        if error:
            st.error(f"Failed to load historical: {error}")
            return pd.DataFrame()
        
        try:
            st.write("📊 Processing historical CSV...")
            df = pd.read_csv(fh, dtype=object)
            st.write(f"Raw shape: {df.shape}")
            
            # Basic cleaning
            df.columns = df.columns.str.strip()
            df['Last Trading Date'] = pd.to_datetime(df['Last Trading Date'], errors='coerce')
            df['Date'] = df['Last Trading Date']  # Alias for merging
            
            # Process numeric columns
            numeric_cols = [
                'High', 'Low', 'Close', 'Volume', 'Value', 'Foreign Buy', 'Foreign Sell',
                'Bid Volume', 'Offer Volume', 'Previous', 'Change', 'Open Price', 'First Trade',
                'Frequency', 'Index Individual', 'Offer', 'Bid', 'Listed Shares', 'Tradeble Shares',
                'Weight For Index', 'Non Regular Volume', 'Change %', 'Typical Price', 'TPxV',
                'VWMA_20D', 'MA20_vol', 'MA5_vol', 'Volume Spike (x)', 'Net Foreign Flow',
                'Bid/Offer Imbalance', 'Money Flow Value', 'Free Float', 'Money Flow Ratio (20D)'
            ]
            
            for col in numeric_cols:
                if col in df.columns:
                    # Clean string values
                    cleaned = df[col].astype(str).str.strip()
                    cleaned = cleaned.str.replace(r'[,\sRp\%]', '', regex=True)
                    df[col] = pd.to_numeric(cleaned, errors='coerce').fillna(0)
            
            # Calculate NFF in Rupiah
            if 'Typical Price' in df.columns:
                df['NFF_Rp'] = df['Net Foreign Flow'] * df['Typical Price']
            else:
                df['NFF_Rp'] = df['Net Foreign Flow'] * df['Close']
            
            # Handle boolean columns
            if 'Unusual Volume' in df.columns:
                df['Unusual Volume'] = df['Unusual Volume'].astype(str).str.strip().str.lower().isin(['spike volume signifikan', 'true'])
            
            # Ensure Sector column
            if 'Sector' in df.columns:
                df['Sector'] = df['Sector'].astype(str).str.strip().fillna('Others')
            else:
                df['Sector'] = 'Others'
            
            st.success(f"✅ Historical loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            st.write("Sample data:")
            st.dataframe(df[['Date', 'Stock Code', 'Close', 'Volume', 'Value', 'Change %']].head())
            
            return df
            
        except Exception as e:
            st.error(f"Error processing historical data: {e}")
            import traceback
            st.code(traceback.format_exc())
            return pd.DataFrame()
    
    @st.cache_data(ttl=3600, show_spinner="🔄 Merging Datasets...")
    def load_merged_data(_self):
        """Intelligent merge of both datasets"""
        st.write("🔄 Starting data merge...")
        
        with st.spinner("Loading datasets..."):
            # Load both datasets
            df_ksei = _self.load_ksei_data()
            df_hist = _self.load_historical_data()
        
        if df_ksei.empty:
            st.error("KSEI dataset is empty")
            return pd.DataFrame()
        
        if df_hist.empty:
            st.error("Historical dataset is empty")
            return pd.DataFrame()
        
        try:
            st.write("🤝 Merging datasets...")
            # Prepare for merge
            df_ksei_m = df_ksei.copy()
            df_hist_m = df_hist.copy()
            
            # Ensure consistent date format
            df_ksei_m['Date'] = pd.to_datetime(df_ksei_m['Date'])
            df_hist_m['Date'] = pd.to_datetime(df_hist_m['Date'])
            
            st.write(f"KSEI dates: {df_ksei_m['Date'].min()} to {df_ksei_m['Date'].max()}")
            st.write(f"Historical dates: {df_hist_m['Date'].min()} to {df_hist_m['Date'].max()}")
            
            # Merge on Date and Stock Code
            merged = pd.merge(
                df_hist_m,
                df_ksei_m[['Date', 'Stock Code', 'Total_chg_Rp', 'Smart_Money_Flow', 
                          'Retail_Flow', 'Institutional_Net', 'Free Float', 'Sector'] + 
                         [c for c in OWNERSHIP_COLS if c in df_ksei_m.columns]],
                on=['Date', 'Stock Code'],
                how='left',
                suffixes=('', '_ksei')
            )
            
            st.write(f"✅ Merge complete: {merged.shape[0]} rows, {merged.shape[1]} columns")
            
            # Show merge statistics
            st.write("📊 Merge Statistics:")
            col1, col2, col3 = st.columns(3)
            with col1:
                matched = merged['Total_chg_Rp'].notna().sum()
                st.metric("Rows with KSEI data", f"{matched:,}")
            
            with col2:
                total = len(merged)
                st.metric("Total Rows", f"{total:,}")
            
            with col3:
                match_percent = (matched / total * 100) if total > 0 else 0
                st.metric("Match Rate", f"{match_percent:.1f}%")
            
            return merged
            
        except Exception as e:
            st.error(f"Merge error: {e}")
            import traceback
            st.code(traceback.format_exc())
            return pd.DataFrame()

# ==============================================================================
# 🎯 MAIN APP (DEBUG VERSION)
# ==============================================================================
def main():
    # DEBUG HEADER
    st.markdown("""
    <div class="header-banner">
        <div class="header-title">🔧 MERGE ANALYTIC - DEBUG MODE</div>
        <div class="header-subtitle">Data Connection Test & System Check</div>
    </div>
    """, unsafe_allow_html=True)
    
    # SIDEBAR FOR CONTROLS
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2910/2910312.png", width=60)
        st.markdown("<h3 style='color:#2B3674;'>🔧 Debug Controls</h3>", unsafe_allow_html=True)
        st.divider()
        
        st.markdown("##### 🚀 Quick Actions")
        if st.button("🔄 Clear Cache & Reload", use_container_width=True, type="primary"):
            st.cache_data.clear()
            st.rerun()
        
        st.divider()
        
        st.markdown("##### ⚙️ Test Options")
        test_mode = st.selectbox(
            "Select Test",
            ["Full Debug", "Test KSEI Only", "Test Historical Only", "Quick Merge Test"]
        )
        
        st.divider()
        
        st.markdown("##### 📁 Folder Info")
        st.code(f"Folder ID: {FOLDER_ID}")
        st.code(f"KSEI File: {FILE_KSEI}")
        st.code(f"Historical File: {FILE_HIST}")
    
    # MAIN DEBUG AREA
    st.markdown("## 🚨 System Diagnostic")
    
    # Initialize loader
    st.write("### 1. Initializing Google Drive Connection")
    loader = DataLoader()
    
    if not loader.service:
        st.error("❌ FAILED: GDrive service not initialized")
        st.markdown("""
        **Possible fixes:**
        1. Check `secrets.toml` format
        2. Verify service account email has access to folder
        3. Check if JSON key is valid
        """)
        return
    
    st.success("✅ Google Drive service initialized")
    
    # List files
    st.write("### 2. Checking Folder Contents")
    files = loader.list_files_in_folder()
    
    if not files:
        st.error("❌ No files found in folder!")
        st.markdown(f"""
        **Troubleshooting:**
        1. Folder ID: `{FOLDER_ID}`
        2. Service account: `streamlit-to-gdrive@stock-analysis-461503.iam.gserviceaccount.com`
        3. Check sharing permissions
        """)
        return
    
    st.success(f"✅ Found {len(files)} file(s)")
    
    # Display files
    with st.expander("📁 View All Files"):
        for file in files:
            st.write(f"**{file['name']}**")
            st.write(f"  • ID: `{file['id']}`")
            st.write(f"  • Size: {file.get('size', 'N/A')} bytes")
            st.write(f"  • Type: {file.get('mimeType', 'N/A')}")
            st.write("---")
    
    # Test downloads based on mode
    st.write("### 3. Testing File Downloads")
    
    if test_mode == "Test KSEI Only" or test_mode == "Full Debug":
        st.markdown(f"#### 📥 Testing KSEI: `{FILE_KSEI}`")
        if st.button(f"Download {FILE_KSEI}", key="dl_ksei"):
            with st.spinner(f"Downloading {FILE_KSEI}..."):
                fh, error = loader.download_file(FILE_KSEI)
                if fh:
                    st.success(f"✅ SUCCESS: {len(fh.getvalue()) / 1024 / 1024:.2f} MB")
                    # Try to read
                    try:
                        df = pd.read_csv(fh, nrows=10)
                        st.write("First 10 rows:")
                        st.dataframe(df)
                    except Exception as e:
                        st.error(f"CSV read error: {e}")
                else:
                    st.error(f"❌ FAILED: {error}")
    
    if test_mode == "Test Historical Only" or test_mode == "Full Debug":
        st.markdown(f"#### 📥 Testing Historical: `{FILE_HIST}`")
        if st.button(f"Download {FILE_HIST}", key="dl_hist"):
            with st.spinner(f"Downloading {FILE_HIST}..."):
                fh, error = loader.download_file(FILE_HIST)
                if fh:
                    st.success(f"✅ SUCCESS: {len(fh.getvalue()) / 1024 / 1024:.2f} MB")
                    # Try to read
                    try:
                        df = pd.read_csv(fh, nrows=10)
                        st.write("First 10 rows:")
                        st.dataframe(df)
                    except Exception as e:
                        st.error(f"CSV read error: {e}")
                else:
                    st.error(f"❌ FAILED: {error}")
    
    # Full merge test
    if test_mode == "Quick Merge Test" or test_mode == "Full Debug":
        st.write("### 4. Testing Full Data Merge")
        
        if st.button("🚀 Run Full Data Pipeline", type="primary"):
            with st.spinner("Running full pipeline..."):
                merged_data = loader.load_merged_data()
            
            if not merged_data.empty:
                st.success(f"🎉 MERGE SUCCESSFUL! {merged_data.shape[0]:,} rows × {merged_data.shape[1]:,} columns")
                
                # Show summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Unique Stocks", merged_data['Stock Code'].nunique())
                with col2:
                    date_range = f"{merged_data['Date'].min().date()} to {merged_data['Date'].max().date()}"
                    st.metric("Date Range", date_range)
                with col3:
                    st.metric("Total Rows", f"{len(merged_data):,}")
                
                # Show sample
                with st.expander("🔍 View Sample Data"):
                    st.dataframe(merged_data.head(20))
                
                # Show columns
                with st.expander("📋 View All Columns"):
                    cols_df = pd.DataFrame({
                        'Column': merged_data.columns,
                        'Type': merged_data.dtypes.astype(str),
                        'Non-Null': merged_data.count().values,
                        'Null %': (merged_data.isnull().sum().values / len(merged_data) * 100).round(1)
                    })
                    st.dataframe(cols_df)
                
                # Launch main dashboard button
                st.markdown("---")
                if st.button("🚀 Launch Main Dashboard", type="primary"):
                    # Switch to main app mode
                    st.session_state.debug_mode = False
                    st.rerun()
            else:
                st.error("❌ Merge failed or returned empty dataset")
    
    # If debug mode is off, show main app
    if 'debug_mode' in st.session_state and not st.session_state.debug_mode:
        show_main_dashboard(loader)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #A3AED0; font-size: 14px;'>"
        "🔧 Debug Mode • Check data connections before analysis"
        "</div>",
        unsafe_allow_html=True
    )

# ==============================================================================
# 🎨 MAIN DASHBOARD (Will be shown after debug)
# ==============================================================================
def show_main_dashboard(loader):
    """Show the main dashboard after debug passes"""
    st.markdown("""
    <div class="header-banner">
        <div class="header-title">🚀 MERGE ANALYTIC KSEI & DAILY TX1Y</div>
        <div class="header-subtitle">Hidden Gem Finder • Institutional Flow Analysis • Multi-dimensional Scoring</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load merged data
    with st.spinner("🚀 Loading merged dataset..."):
        df_merged = loader.load_merged_data()
    
    if df_merged.empty:
        st.error("Failed to load data. Please check debug mode first.")
        if st.button("🔧 Enter Debug Mode"):
            st.session_state.debug_mode = True
            st.rerun()
        return
    
    # Continue with the rest of your dashboard...
    # ... [Insert the rest of your dashboard code here] ...
    
    st.success("✅ Dashboard loaded successfully!")
    st.write(f"Data: {df_merged.shape[0]:,} rows × {df_merged.shape[1]:,} columns")

# ==============================================================================
# 🚀 RUN APP
# ==============================================================================
if __name__ == "__main__":
    # Initialize session state
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = True  # Start in debug mode
    
    # Run appropriate mode
    if st.session_state.debug_mode:
        main()
    else:
        # Initialize loader and show dashboard
        loader = DataLoader()
        if loader.service:
            show_main_dashboard(loader)
        else:
            st.error("Failed to initialize. Entering debug mode...")
            st.session_state.debug_mode = True
            st.rerun()
