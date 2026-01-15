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
import base64
import ast
import re

# Google Drive imports
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ==============================================================================
# ⚙️ PAGE CONFIG & CSS
# ==============================================================================
st.set_page_config(
    page_title="MERGE ANALYTIC KSEI & DAILY TX1Y",
    layout="wide",
    page_icon="📊",
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
# 📦 DATA LOADER CLASS - FIXED VERSION
# ==============================================================================
class DataLoader:
    def __init__(self):
        self.service = None
        self.initialize_gdrive()
    
    def parse_creds_from_secrets(self, creds_data):
        """Robust parsing of credentials from Streamlit secrets"""
        try:
            # Case 1: Already a dict (local development)
            if isinstance(creds_data, dict):
                return creds_data
            
            # Case 2: String that needs parsing
            if isinstance(creds_data, str):
                # Clean the string
                creds_str = creds_data.strip()
                
                # Remove wrapping quotes
                if (creds_str.startswith("'") and creds_str.endswith("'")) or \
                   (creds_str.startswith('"') and creds_str.endswith('"')):
                    creds_str = creds_str[1:-1]
                
                # Remove triple quotes if present
                if creds_str.startswith("'''") and creds_str.endswith("'''"):
                    creds_str = creds_str[3:-3]
                elif creds_str.startswith('"""') and creds_str.endswith('"""'):
                    creds_str = creds_str[3:-3]
                
                # Fix escaped characters
                creds_str = creds_str.replace('\\n', '\n').replace('\\\\n', '\n')
                creds_str = creds_str.replace('\\"', '"').replace("\\'", "'")
                
                # Try JSON parsing first
                try:
                    return json.loads(creds_str)
                except json.JSONDecodeError:
                    # Try ast.literal_eval for Python-style strings
                    try:
                        return ast.literal_eval(creds_str)
                    except:
                        # Last resort: regex extraction
                        import re
                        json_match = re.search(r'\{.*\}', creds_str, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(0)
                            # Fix newlines in the JSON string
                            json_str = json_str.replace('\n', '\\n')
                            return json.loads(json_str)
                        else:
                            raise ValueError("Could not parse credentials string")
            
            # Case 3: Base64 encoded
            if isinstance(creds_data, str) and len(creds_data) > 100 and '=' in creds_data[-2:]:
                try:
                    decoded = base64.b64decode(creds_data).decode('utf-8')
                    return json.loads(decoded)
                except:
                    pass
            
            raise ValueError(f"Unknown credentials format: {type(creds_data)}")
            
        except Exception as e:
            st.error(f"Credential parsing error: {e}")
            return None
    
    def initialize_gdrive(self):
        """Initialize Google Drive service with Streamlit secrets"""
        try:
            # Get credentials from secrets
            if "gcp_service_account" not in st.secrets:
                st.error("❌ 'gcp_service_account' not found in secrets.toml")
                return False
            
            creds_data = st.secrets["gcp_service_account"]
            
            # Parse credentials
            creds_json = self.parse_creds_from_secrets(creds_data)
            if creds_json is None:
                return False
            
            # Create credentials object
            creds = Credentials.from_service_account_info(
                creds_json, 
                scopes=['https://www.googleapis.com/auth/drive.readonly']
            )
            
            # Build service
            self.service = build('drive', 'v3', credentials=creds, cache_discovery=False)
            
            # Test the service
            try:
                about = self.service.about().get(fields="user").execute()
                st.success(f"✅ Google Drive authenticated: {about.get('user', {}).get('emailAddress', 'Service Account')}")
            except Exception as test_err:
                st.warning(f"⚠️ Auth test: {test_err}")
                # Continue anyway
            
            return True
            
        except Exception as e:
            st.error(f"❌ GDrive Initialization Error: {type(e).__name__}: {str(e)}")
            return False
    
    def list_files_in_folder(self):
        """List all files in the folder"""
        if not self.service:
            return []
        
        try:
            query = f"'{FOLDER_ID}' in parents and trashed=false"
            results = self.service.files().list(
                q=query, 
                fields="files(id, name, mimeType, size, modifiedTime)",
                pageSize=100
            ).execute()
            
            return results.get('files', [])
            
        except Exception as e:
            st.error(f"Error listing files: {e}")
            return []
    
    def download_file(self, file_name):
        """Download file from Google Drive"""
        if not self.service:
            return None, "Service not initialized"
        
        try:
            # Search for file
            query = f"'{FOLDER_ID}' in parents and name='{file_name}' and trashed=false"
            results = self.service.files().list(q=query, fields="files(id, name)", pageSize=1).execute()
            items = results.get('files', [])
            
            if not items:
                return None, f"File '{file_name}' not found"
            
            file_id = items[0]['id']
            
            # Download file
            request = self.service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            
            while not done:
                status, done = downloader.next_chunk()
            
            fh.seek(0)
            return fh, None
            
        except Exception as e:
            return None, f"Download error: {e}"
    
    @st.cache_data(ttl=3600, show_spinner="🔄 Loading KSEI Data...")
    def load_ksei_data(_self):
        """Load and process KSEI ownership data"""
        fh, error = _self.download_file(FILE_KSEI)
        if error:
            st.error(f"KSEI: {error}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(fh, dtype=object)
            
            # Basic cleaning
            df.columns = df.columns.str.strip()
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
            # Filter recent data
            df = df[df['Date'].dt.year >= 2024].copy()
            
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
            
            return df
            
        except Exception as e:
            st.error(f"Error processing KSEI data: {e}")
            return pd.DataFrame()
    
    @st.cache_data(ttl=3600, show_spinner="🔄 Loading Historical Data...")
    def load_historical_data(_self):
        """Load and process 1-year historical data"""
        fh, error = _self.download_file(FILE_HIST)
        if error:
            st.error(f"Historical: {error}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(fh, dtype=object)
            
            # Basic cleaning
            df.columns = df.columns.str.strip()
            df['Last Trading Date'] = pd.to_datetime(df['Last Trading Date'], errors='coerce')
            df['Date'] = df['Last Trading Date']
            
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
            
            return df
            
        except Exception as e:
            st.error(f"Error processing historical data: {e}")
            return pd.DataFrame()
    
    @st.cache_data(ttl=3600, show_spinner="🔄 Merging Datasets...")
    def load_merged_data(_self):
        """Intelligent merge of both datasets"""
        # Load both datasets
        df_ksei = _self.load_ksei_data()
        df_hist = _self.load_historical_data()
        
        if df_ksei.empty or df_hist.empty:
            return pd.DataFrame()
        
        try:
            # Prepare for merge
            df_ksei_m = df_ksei.copy()
            df_hist_m = df_hist.copy()
            
            # Ensure consistent date format
            df_ksei_m['Date'] = pd.to_datetime(df_ksei_m['Date'])
            df_hist_m['Date'] = pd.to_datetime(df_hist_m['Date'])
            
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
            
            # Forward fill KSEI data for continuity
            ksei_cols = ['Total_chg_Rp', 'Smart_Money_Flow', 'Retail_Flow', 'Institutional_Net', 'Free Float']
            merged = merged.sort_values(['Stock Code', 'Date'])
            
            for col in ksei_cols:
                if col in merged.columns:
                    merged[col] = merged.groupby('Stock Code')[col].ffill()
            
            # Calculate derived metrics
            merged['Price_Change_1D'] = merged.groupby('Stock Code')['Close'].pct_change()
            merged['Volume_Change_1D'] = merged.groupby('Stock Code')['Volume'].pct_change()
            
            if 'Money Flow Value' in merged.columns:
                merged['MF_Strength'] = merged['Money Flow Value'] / merged['Value'].replace(0, 1)
            
            return merged
            
        except Exception as e:
            st.error(f"Merge error: {e}")
            return pd.DataFrame()

# ==============================================================================
# 🎨 SIMPLE DASHBOARD
# ==============================================================================
def show_simple_dashboard(loader):
    """Show a simple dashboard that works"""
    st.markdown("""
    <div class="header-banner">
        <div class="header-title">🚀 MERGE ANALYTIC KSEI & DAILY TX1Y</div>
        <div class="header-subtitle">Simple & Working Version</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2910/2910312.png", width=60)
        st.markdown("<h3 style='color:#2B3674;'>📊 Dashboard</h3>", unsafe_allow_html=True)
        st.divider()
        
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.divider()
        st.markdown("**Data Status:**")
        
        # Quick check
        files = loader.list_files_in_folder()
        if files:
            st.success(f"✅ {len(files)} files available")
            with st.expander("View Files"):
                for f in files:
                    st.write(f"📄 {f['name']}")
        else:
            st.error("❌ No files found")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📈 Data Loader", "📊 Analysis", "🔍 Explore"])
    
    with tab1:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📥 Data Loading Test</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Test Load KSEI", type="primary"):
                with st.spinner("Loading KSEI data..."):
                    df_ksei = loader.load_ksei_data()
                    if not df_ksei.empty:
                        st.success(f"✅ KSEI loaded: {df_ksei.shape[0]:,} rows")
                        st.dataframe(df_ksei[['Date', 'Stock Code', 'Price', 'Total_chg_Rp']].head(10))
                        st.metric("Date Range", f"{df_ksei['Date'].min().date()} to {df_ksei['Date'].max().date()}")
                    else:
                        st.error("Failed to load KSEI")
        
        with col2:
            if st.button("Test Load Historical"):
                with st.spinner("Loading historical data..."):
                    df_hist = loader.load_historical_data()
                    if not df_hist.empty:
                        st.success(f"✅ Historical loaded: {df_hist.shape[0]:,} rows")
                        st.dataframe(df_hist[['Date', 'Stock Code', 'Close', 'Volume', 'Value']].head(10))
                        st.metric("Date Range", f"{df_hist['Date'].min().date()} to {df_hist['Date'].max().date()}")
                    else:
                        st.error("Failed to load historical")
        
        # Merge test
        st.markdown("---")
        if st.button("🚀 Merge Both Datasets", type="primary"):
            with st.spinner("Merging datasets..."):
                df_merged = loader.load_merged_data()
                if not df_merged.empty:
                    st.success(f"🎉 MERGE SUCCESS! {df_merged.shape[0]:,} rows × {df_merged.shape[1]:,} columns")
                    
                    # Show stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Unique Stocks", df_merged['Stock Code'].nunique())
                    with col2:
                        st.metric("Date Range", f"{df_merged['Date'].min().date()} to {df_merged['Date'].max().date()}")
                    with col3:
                        match_rate = (df_merged['Total_chg_Rp'].notna().sum() / len(df_merged) * 100)
                        st.metric("KSEI Match", f"{match_rate:.1f}%")
                    
                    # Save to session for other tabs
                    st.session_state.df_merged = df_merged
                    st.session_state.data_loaded = True
                    
                    # Show sample
                    with st.expander("View Sample Data"):
                        st.dataframe(df_merged.head(20))
                else:
                    st.error("Merge failed")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📊 Quick Analysis</div>', unsafe_allow_html=True)
        
        if 'data_loaded' in st.session_state and st.session_state.data_loaded:
            df = st.session_state.df_merged
            
            # Date selector
            latest_dates = sorted(df['Date'].unique(), reverse=True)[:10]
            selected_date = st.selectbox(
                "Select Date",
                options=latest_dates,
                format_func=lambda x: x.strftime('%d %b %Y'),
                index=0
            )
            
            if selected_date:
                daily = df[df['Date'] == selected_date]
                
                # Top metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    top_gainer = daily.nlargest(1, 'Change %')
                    if not top_gainer.empty:
                        st.metric("📈 Top Gainer", 
                                 top_gainer.iloc[0]['Stock Code'],
                                 delta=f"{top_gainer.iloc[0]['Change %']:.2f}%")
                
                with col2:
                    top_value = daily.nlargest(1, 'Value')
                    if not top_value.empty:
                        st.metric("💰 Top Value", 
                                 f"Rp {top_value.iloc[0]['Value']/1e9:.1f}B",
                                 top_value.iloc[0]['Stock Code'])
                
                with col3:
                    if 'Smart_Money_Flow' in daily.columns:
                        top_smart = daily.nlargest(1, 'Smart_Money_Flow')
                        if not top_smart.empty:
                            st.metric("🧠 Top Smart Money", 
                                     top_smart.iloc[0]['Stock Code'],
                                     delta=f"Rp {top_smart.iloc[0]['Smart_Money_Flow']/1e9:.1f}B")
                
                # Charts
                st.markdown("#### 📈 Top 10 by Smart Money Flow")
                if 'Smart_Money_Flow' in daily.columns:
                    top_smart_10 = daily.nlargest(10, 'Smart_Money_Flow')
                    if not top_smart_10.empty:
                        fig = px.bar(
                            top_smart_10,
                            x='Stock Code',
                            y='Smart_Money_Flow',
                            color='Smart_Money_Flow',
                            color_continuous_scale='Greens',
                            title=f"Smart Money Flow - {selected_date.strftime('%d %b %Y')}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👆 Please load and merge data first in the Data Loader tab")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🔍 Data Explorer</div>', unsafe_allow_html=True)
        
        if 'data_loaded' in st.session_state and st.session_state.data_loaded:
            df = st.session_state.df_merged
            
            # Stock selector
            stocks = sorted(df['Stock Code'].unique())
            selected_stock = st.selectbox("Select Stock", stocks)
            
            if selected_stock:
                stock_data = df[df['Stock Code'] == selected_stock].sort_values('Date')
                
                if not stock_data.empty:
                    # Latest info
                    latest = stock_data.iloc[-1]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Price", f"Rp {latest.get('Close', 0):,.0f}")
                    with col2:
                        st.metric("Change", f"{latest.get('Change %', 0):.2f}%")
                    with col3:
                        if 'Smart_Money_Flow' in latest:
                            st.metric("Smart Money", f"Rp {latest['Smart_Money_Flow']/1e9:.1f}B")
                    with col4:
                        st.metric("Sector", latest.get('Sector', 'N/A'))
                    
                    # Price chart
                    st.markdown("#### 📈 Price History")
                    fig = px.line(
                        stock_data,
                        x='Date',
                        y='Close',
                        title=f"{selected_stock} Price History"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show data
                    with st.expander("View Raw Data"):
                        st.dataframe(stock_data)
        else:
            st.info("👆 Please load and merge data first")
        
        st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# 🚀 MAIN APP
# ==============================================================================
def main():
    # Initialize loader
    loader = DataLoader()
    
    if not loader.service:
        st.error("❌ Failed to initialize Google Drive service")
        st.markdown("""
        ### 🔧 Troubleshooting Steps:
        
        1. **Check secrets.toml** - Make sure it's in `.streamlit/secrets.toml`
        2. **Format should be:**
        ```toml
        gcp_service_account = '''
        {
          "type": "service_account",
          "project_id": "...",
          "private_key_id": "...",
          "private_key": "-----BEGIN PRIVATE KEY-----\\n...\\n-----END PRIVATE KEY-----\\n",
          ...
        }
        '''
        ```
        
        3. **Verify sharing** - Folder is shared with:  
           `streamlit-to-gdrive@stock-analysis-461503.iam.gserviceaccount.com`
        
        4. **File names** - Ensure they exist:  
           • `KSEI_Shareholder_Processed.csv`  
           • `Kompilasi_Data_1Tahun.csv`
        """)
        return
    
    # Show dashboard
    show_simple_dashboard(loader)

# ==============================================================================
# 🚀 RUN APP
# ==============================================================================
if __name__ == "__main__":
    main()
