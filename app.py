# ==============================================================================
# 🚀 HIDDEN GEM FINDER v3.0 - ENTERPRISE EDITION
# ==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import time
import json
import base64
import logging
from datetime import datetime, timedelta
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Google Drive imports
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ==============================================================================
# ⚙️ CONFIGURATION & CONSTANTS
# ==============================================================================
class Config:
    """Centralized configuration"""
    FOLDER_ID = "1hX2jwUrAgi4Fr8xkcFWjCW6vbk6lsIlP"
    FILE_KSEI = "KSEI_Shareholder_Processed.csv"
    FILE_HIST = "Kompilasi_Data_1Tahun.csv"
    
    # Ownership categories
    OWNERSHIP_COLS = [
        'Local IS', 'Local CP', 'Local PF', 'Local IB', 'Local ID', 'Local MF', 
        'Local SC', 'Local FD', 'Local OT',
        'Foreign IS', 'Foreign CP', 'Foreign PF', 'Foreign IB', 'Foreign ID', 
        'Foreign MF', 'Foreign SC', 'Foreign FD', 'Foreign OT'
    ]
    
    # Performance settings
    MAX_WORKERS = 4
    CACHE_TTL = 3600
    MAX_STOCKS_ANALYZED = 200
    
    # Scoring weights
    SCORE_WEIGHTS = {
        'smart_money': 0.40,
        'technical': 0.30,
        'fundamental': 0.15,
        'volatility': 0.15
    }
    
    # Market regimes
    REGIME_MULTIPLIERS = {
        'bull': 1.1,
        'neutral': 1.0,
        'bear': 0.9,
        'correction': 0.85
    }

# ==============================================================================
# 🎨 UI CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="💎 HIDDEN GEM FINDER v3.0 - Enterprise",
    layout="wide",
    page_icon="💎",
    initial_sidebar_state="expanded"
)

# --- ENHANCED CSS WITH ANIMATIONS ---
st.markdown("""
<style>
    /* KSEI Professional Theme */
    :root {
        --primary-color: #4318FF;
        --primary-light: #868CFF;
        --background-color: #F4F7FE;
        --card-background: #FFFFFF;
        --text-color: #2B3674;
        --text-secondary: #A3AED0;
        --success-color: #05CD99;
        --warning-color: #FFB547;
        --danger-color: #EE5D50;
        --font: 'DM Sans', sans-serif;
    }
    
    .stApp {
        background-color: var(--background-color);
        color: var(--text-color);
        font-family: var(--font);
    }
    
    /* Enhanced Header */
    .header-gradient {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-light) 100%);
        border-radius: 20px;
        padding: 30px 40px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0px 18px 40px rgba(112, 144, 176, 0.2);
        animation: fadeInDown 0.8s ease-out;
    }
    
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .header-title { 
        font-size: 2.5rem; 
        font-weight: 800; 
        margin-bottom: 10px;
        background: linear-gradient(90deg, #FFFFFF 0%, #F0F0FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .header-subtitle { 
        font-size: 1.1rem; 
        font-weight: 500; 
        opacity: 0.9;
    }
    
    /* Enhanced Cards with Glassmorphism */
    .css-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 28px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0px 18px 40px rgba(112, 144, 176, 0.12);
        margin-bottom: 24px;
        transition: all 0.3s ease;
        animation: slideUp 0.5s ease-out;
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .css-card:hover {
        transform: translateY(-5px);
        box-shadow: 0px 25px 50px rgba(112, 144, 176, 0.15);
    }
    
    .card-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-color);
        margin-bottom: 24px;
        border-bottom: 2px solid var(--primary-color);
        padding-bottom: 12px;
    }
    
    /* Enhanced Metrics */
    .metric-card {
        background: linear-gradient(135deg, var(--card-background) 0%, #F8F9FF 100%);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(67, 24, 255, 0.1);
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .metric-card:hover {
        border-color: var(--primary-color);
        box-shadow: 0 10px 25px rgba(67, 24, 255, 0.1);
    }
    
    /* Enhanced Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: transparent;
        padding: 4px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        padding: 0 24px;
        background-color: rgba(255, 255, 255, 0.7);
        border-radius: 10px;
        font-weight: 600;
        border: 1px solid rgba(67, 24, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(67, 24, 255, 0.05);
        border-color: var(--primary-color);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, var(--primary-color) 0%, var(--primary-light) 100%);
        color: white !important;
        border-color: var(--primary-color);
        box-shadow: 0 4px 12px rgba(67, 24, 255, 0.2);
    }
    
    /* Enhanced Buttons */
    div.stButton > button {
        background: linear-gradient(90deg, var(--primary-color) 0%, var(--primary-light) 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        width: 100%;
        box-shadow: 0px 4px 15px rgba(67, 24, 255, 0.2);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0px 8px 25px rgba(67, 24, 255, 0.3);
    }
    
    div.stButton > button:after {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.2);
        transform: translate(-50%, -50%);
        transition: width 0.3s, height 0.3s;
    }
    
    div.stButton > button:active:after {
        width: 200px;
        height: 200px;
    }
    
    /* Badges */
    .gem-badge {
        background: linear-gradient(90deg, #FFD700 0%, #FFA500 100%);
        color: #2B3674;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 12px;
        display: inline-block;
        margin: 2px;
        box-shadow: 0 2px 8px rgba(255, 168, 0, 0.2);
    }
    
    .signal-buy { 
        background: linear-gradient(90deg, #05CD99 0%, #00B894 100%);
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 12px;
    }
    
    .signal-sell { 
        background: linear-gradient(90deg, #EE5D50 0%, #FF6B6B 100%);
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 12px;
    }
    
    .signal-neutral { 
        background: linear-gradient(90deg, #A3AED0 0%, #95A5A6 100%);
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 12px;
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--primary-color) 0%, var(--primary-light) 100%);
        border-radius: 10px;
    }
    
    /* Sidebar enhancements */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FFFFFF 0%, #F8F9FF 100%);
        box-shadow: 14px 14px 40px rgba(112, 144, 176, 0.12);
    }
    
    /* Data tables */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(67, 24, 255, 0.1);
    }
    
    /* Tooltips */
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted var(--text-secondary);
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: var(--text-color);
        color: white;
        text-align: center;
        border-radius: 6px;
        padding: 8px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 12px;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Loading animation */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-green { background-color: var(--success-color); }
    .status-yellow { background-color: var(--warning-color); }
    .status-red { background-color: var(--danger-color); }
    .status-blue { background-color: var(--primary-color); }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 🛠️ UTILITY FUNCTIONS
# ==============================================================================
class PerformanceMonitor:
    """Performance monitoring and timing utilities"""
    
    @staticmethod
    def time_execution(func):
        """Decorator to time function execution"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            # Store in session state for display
            if 'performance_metrics' not in st.session_state:
                st.session_state.performance_metrics = {}
            
            st.session_state.performance_metrics[func.__name__] = {
                'elapsed': elapsed,
                'timestamp': datetime.now().isoformat()
            }
            
            return result
        return wrapper
    
    @staticmethod
    def display_performance_metrics():
        """Display performance metrics in sidebar"""
        if 'performance_metrics' in st.session_state and st.session_state.performance_metrics:
            with st.sidebar.expander("⚡ Performance Metrics"):
                metrics = st.session_state.performance_metrics
                for func_name, data in metrics.items():
                    st.metric(f"{func_name}", f"{data['elapsed']:.2f}s")


class ErrorHandler:
    """Comprehensive error handling and logging"""
    
    def __init__(self):
        self.logger = logging.getLogger('GemFinder')
        self.logger.setLevel(logging.ERROR)
        
        # Create file handler
        fh = logging.FileHandler(f'gem_finder_errors_{datetime.now().strftime("%Y%m%d")}.log')
        fh.setLevel(logging.ERROR)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(fh)
    
    def safe_execute(self, func, *args, fallback_value=None, **kwargs):
        """Safely execute function with error handling"""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            
            # Show user-friendly error message
            st.toast(f"⚠️ {func.__name__} encountered an error", icon="⚠️")
            
            # Return fallback value if provided
            if fallback_value is not None:
                return fallback_value
            
            # Default fallback based on function type
            if func.__name__ == 'calculate_gem_score':
                return {'total_score': 0, 'signal': 'ERROR', 'signal_color': 'neutral'}
            elif func.__name__ == 'find_top_gems':
                return pd.DataFrame()
            else:
                return None


# ==============================================================================
# 📦 ENHANCED DATA LOADER - SIMPLIFIED WORKING VERSION
# ==============================================================================
class EnhancedDataLoader:
    """Enhanced data loader with validation and monitoring - SIMPLIFIED WORKING VERSION"""
    
    def __init__(self):
        self.service = None
        self.error_handler = ErrorHandler()
        self.initialize_gdrive()
    
    def initialize_gdrive(self):
    """Simple Google Drive initialization - FIXED VERSION"""
    try:
        if "gcp_service_account" not in st.secrets:
            st.error("❌ 'gcp_service_account' not found in secrets.toml")
            return False
        
        # Get credentials data
        creds_data = st.secrets["gcp_service_account"]
        
        # Parse credentials
        if isinstance(creds_data, dict):
            creds_json = creds_data
        elif isinstance(creds_data, str):
            # Clean string - remove surrounding quotes if present
            creds_str = creds_data.strip()
            
            # For multi-line JSON in TOML, we might have triple quotes
            if creds_str.startswith("'''") and creds_str.endswith("'''"):
                creds_str = creds_str[3:-3]
            elif creds_str.startswith('"""') and creds_str.endswith('"""'):
                creds_str = creds_str[3:-3]
            elif creds_str.startswith("'") and creds_str.endswith("'"):
                creds_str = creds_str[1:-1]
            elif creds_str.startswith('"') and creds_str.endswith('"'):
                creds_str = creds_str[1:-1]
            
            # DEBUG: Show what we're parsing
            st.write("🔍 Parsing JSON string...")
            
            # Parse JSON directly - \n should be preserved as is
            creds_json = json.loads(creds_str)
            
            # After parsing, if private_key has \n, keep it as is
            # JSON parser will convert \n to actual newline character
            if 'private_key' in creds_json:
                # It should already be correct after json.loads()
                pass
        else:
            st.error(f"❌ Unknown credentials type: {type(creds_data)}")
            return False
        
        st.success(f"✅ Parsed credentials for: {creds_json.get('client_email', 'Unknown')}")
        
        # Create credentials
        creds = Credentials.from_service_account_info(
            creds_json,
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        
        # Build service
        self.service = build('drive', 'v3', credentials=creds, cache_discovery=False)
        
        # Test connection
        about = self.service.about().get(fields="user").execute()
        st.success(f"✅ Google Drive connected: {about.get('user', {}).get('emailAddress', 'Service Account')}")
        return True
        
    except json.JSONDecodeError as e:
        st.error(f"❌ JSON Parse Error: {e}")
        st.error(f"Error at position: {e.pos}")
        
        # Show more context
        if 'creds_str' in locals():
            start = max(0, e.pos - 50)
            end = min(len(creds_str), e.pos + 50)
            st.code(f"...{creds_str[start:end]}...", language="text")
        
        return False
    except Exception as e:
        st.error(f"❌ Google Drive Error: {type(e).__name__}: {str(e)}")
        return False
    
    # ========== KEEP ALL OTHER METHODS THE SAME ==========
    # JANGAN ganti method-method di bawah ini, hanya method initialize_gdrive() saja
    # yang di atas ini
    
    def list_files_in_folder(self):
        """List all files in the folder"""
        if not self.service:
            return []
        
        try:
            query = f"'{Config.FOLDER_ID}' in parents and trashed=false"
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
            query = f"'{Config.FOLDER_ID}' in parents and name='{file_name}' and trashed=false"
            results = self.service.files().list(q=query, fields="files(id, name)", pageSize=1).execute()
            items = results.get('files', [])
            
            if not items:
                return None, f"File '{file_name}' not found"
            
            file_id = items[0]['id']
            request = self.service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            
            while not done:
                _, done = downloader.next_chunk()
            
            fh.seek(0)
            return fh, None
            
        except Exception as e:
            return None, f"Download error: {e}"
    
    @PerformanceMonitor.time_execution
    @st.cache_data(ttl=Config.CACHE_TTL, show_spinner="📅 Loading Monthly KSEI Data...")
    def load_ksei_data(_self):
        """Load and process MONTHLY KSEI ownership data"""
        return _self.error_handler.safe_execute(
            _self._load_ksei_data_internal,
            fallback_value=pd.DataFrame()
        )
    
    def _load_ksei_data_internal(self):
        """Internal KSEI data loading"""
        fh, error = self.download_file(Config.FILE_KSEI)
        if error:
            raise Exception(f"KSEI Download: {error}")
        
        df = pd.read_csv(fh, dtype=object)
        df.columns = df.columns.str.strip()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df[df['Date'].dt.year >= 2023].copy()
        
        # Process numeric columns
        numeric_cols = Config.OWNERSHIP_COLS + [f"{col}_chg" for col in Config.OWNERSHIP_COLS] + \
                      [f"{col}_chg_Rp" for col in Config.OWNERSHIP_COLS] + \
                      ['Price', 'Free Float', 'Total_Local', 'Total_Foreign']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(',', ''), 
                    errors='coerce'
                ).fillna(0)
        
        # Calculate derived metrics
        local_cols = [c for c in [f"{col}_chg_Rp" for col in Config.OWNERSHIP_COLS] 
                     if 'Local' in c and c in df.columns]
        foreign_cols = [c for c in [f"{col}_chg_Rp" for col in Config.OWNERSHIP_COLS] 
                       if 'Foreign' in c and c in df.columns]
        
        df['Total_Local_chg_Rp'] = df[local_cols].sum(axis=1) if local_cols else 0
        df['Total_Foreign_chg_Rp'] = df[foreign_cols].sum(axis=1) if foreign_cols else 0
        df['Total_chg_Rp'] = df['Total_Local_chg_Rp'] + df['Total_Foreign_chg_Rp']
        
        # Smart money calculation
        smart_money_cols = [
            'Foreign IS_chg_Rp', 'Foreign IB_chg_Rp', 'Foreign PF_chg_Rp',
            'Local IS_chg_Rp', 'Local PF_chg_Rp', 'Local MF_chg_Rp', 'Local IB_chg_Rp'
        ]
        smart_money_cols = [c for c in smart_money_cols if c in df.columns]
        df['Smart_Money_Flow'] = df[smart_money_cols].sum(axis=1) if smart_money_cols else 0
        
        # Retail calculation
        retail_cols = ['Local ID_chg_Rp']
        retail_cols = [c for c in retail_cols if c in df.columns]
        df['Retail_Flow'] = df[retail_cols].sum(axis=1) if retail_cols else 0
        
        df['Institutional_Net'] = df['Smart_Money_Flow'] - df['Retail_Flow']
        df['Stock Code'] = df['Code']
        
        if 'Free Float' in df.columns:
            df['Ownership_Concentration'] = 100 - df['Free Float']
        
        return df
    
    @PerformanceMonitor.time_execution
    @st.cache_data(ttl=Config.CACHE_TTL, show_spinner="📈 Loading Daily Historical Data...")
    def load_historical_data(_self):
        """Load and process DAILY historical data"""
        return _self.error_handler.safe_execute(
            _self._load_historical_data_internal,
            fallback_value=pd.DataFrame()
        )
    
    def _load_historical_data_internal(self):
        """Internal historical data loading"""
        if not self.service:
            return pd.DataFrame()
        
        fh, error = self.download_file(Config.FILE_HIST)
        if error:
            raise Exception(f"Historical Download: {error}")
        
        df = pd.read_csv(fh, dtype=object)
        df.columns = df.columns.str.strip()
        df['Last Trading Date'] = pd.to_datetime(df['Last Trading Date'], errors='coerce')
        df['Date'] = df['Last Trading Date']
        
        # Numeric columns to process
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
        
        # Calculate derived metrics
        if 'Typical Price' in df.columns:
            df['NFF_Rp'] = df['Net Foreign Flow'] * df['Typical Price']
        else:
            df['NFF_Rp'] = df['Net Foreign Flow'] * df['Close']
        
        if 'Unusual Volume' in df.columns:
            df['Unusual Volume'] = df['Unusual Volume'].astype(str).str.strip().str.lower().isin(
                ['spike volume signifikan', 'true', 'yes']
            )
        
        if 'Sector' in df.columns:
            df['Sector'] = df['Sector'].astype(str).str.strip().fillna('Others')
        else:
            df['Sector'] = 'Others'
        
        # Technical indicators
        if 'Close' in df.columns:
            # Moving averages
            df['Price_MA20'] = df.groupby('Stock Code')['Close'].transform(
                lambda x: x.rolling(20, min_periods=5).mean()
            )
            df['Price_MA50'] = df.groupby('Stock Code')['Close'].transform(
                lambda x: x.rolling(50, min_periods=10).mean()
            )
            
            # RSI calculation
            def calculate_rsi(prices, period=14):
                if len(prices) < period + 1:
                    return pd.Series([50] * len(prices), index=prices.index)
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
                loss = loss.replace(0, np.nan)
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                return rsi.fillna(50).clip(0, 100)
            
            df['RSI_14'] = df.groupby('Stock Code')['Close'].transform(calculate_rsi)
        
        return df
    
    @PerformanceMonitor.time_execution
    @st.cache_data(ttl=Config.CACHE_TTL, show_spinner="🔄 Merging Monthly KSEI + Daily Data...")
    def load_merged_data(_self):
        """Intelligent merge of MONTHLY KSEI + DAILY historical data"""
        return _self.error_handler.safe_execute(
            _self._merge_data_internal,
            fallback_value=pd.DataFrame()
        )
    
    def _merge_data_internal(self):
        """Internal data merging logic"""
        df_ksei = self.load_ksei_data()
        df_hist = self.load_historical_data()
        
        if df_ksei.empty or df_hist.empty:
            return pd.DataFrame()
        
        # Prepare data
        df_ksei_m = df_ksei.copy()
        df_hist_m = df_hist.copy()
        
        df_ksei_m['Date'] = pd.to_datetime(df_ksei_m['Date'])
        df_hist_m['Date'] = pd.to_datetime(df_hist_m['Date'])
        
        # Create complete date-stock grid
        all_dates = pd.date_range(
            start=df_hist_m['Date'].min(), 
            end=df_hist_m['Date'].max(), 
            freq='D'
        )
        all_stocks = pd.Series(pd.unique(df_hist_m['Stock Code'])).dropna()
        
        date_stock_grid = pd.MultiIndex.from_product(
            [all_dates, all_stocks],
            names=['Date', 'Stock Code']
        )
        complete_grid = pd.DataFrame(index=date_stock_grid).reset_index()
        
        # Merge daily data first
        merged = pd.merge(
            complete_grid,
            df_hist_m,
            on=['Date', 'Stock Code'],
            how='left'
        ).sort_values(['Stock Code', 'Date'])
        
        # KSEI columns to forward fill
        ksei_cols = [
            'Total_chg_Rp', 'Smart_Money_Flow', 'Retail_Flow', 
            'Institutional_Net', 'Free Float', 'Sector',
            'Ownership_Concentration', 'Price'
        ]
        ksei_cols = [col for col in ksei_cols if col in df_ksei_m.columns]
        
        # Forward fill monthly KSEI data to daily
        for col in ksei_cols:
            temp_ksei = df_ksei_m[['Date', 'Stock Code', col]].copy()
            temp_ksei = temp_ksei.dropna(subset=[col])
            
            raw_col_name = f'{col}_ksei_raw'
            temp_ksei = temp_ksei.rename(columns={col: raw_col_name})
            
            merged = pd.merge(
                merged,
                temp_ksei,
                on=['Date', 'Stock Code'],
                how='left'
            )
            
            merged[col] = merged.groupby('Stock Code')[raw_col_name].ffill()
            
            if raw_col_name in merged.columns:
                merged = merged.drop(columns=[raw_col_name])
        
        # Fill missing Close with KSEI Price
        if 'Price' in merged.columns and 'Close' in merged.columns:
            merged['Close'] = merged['Close'].fillna(merged['Price'])
        
        # Calculate derived metrics
        merged['Price_Change_1D'] = merged.groupby('Stock Code')['Close'].pct_change()
        
        if 'Volume' in merged.columns:
            merged['Volume_Change_1D'] = merged.groupby('Stock Code')['Volume'].pct_change()
        
        if 'Money Flow Value' in merged.columns and 'Value' in merged.columns:
            merged['MF_Strength'] = merged['Money Flow Value'] / merged['Value'].replace(0, 1)
        
        # Remove rows with no data
        merged = merged.dropna(subset=['Close', 'Smart_Money_Flow'], how='all')
        
        # Add data type flags
        merged['Has_KSEI_Data'] = merged['Smart_Money_Flow'].notna()
        merged['Has_Daily_Data'] = merged['Close'].notna()
        
        return merged
    
    def validate_data_integrity(self, df_merged):
        """Comprehensive data validation"""
        checks = {
            "total_rows": len(df_merged),
            "date_range": f"{df_merged['Date'].min().date()} to {df_merged['Date'].max().date()}",
            "unique_stocks": df_merged['Stock Code'].nunique(),
            "null_percentage": f"{(df_merged.isnull().sum().sum() / (len(df_merged) * len(df_merged.columns)) * 100):.1f}%",
            "ksei_coverage": f"{(df_merged['Smart_Money_Flow'].notna().sum() / len(df_merged) * 100):.1f}%",
            "daily_coverage": f"{(df_merged['Close'].notna().sum() / len(df_merged) * 100):.1f}%",
            "duplicates": df_merged.duplicated(subset=['Date', 'Stock Code']).sum(),
            "data_consistency": {
                'price_positive': (df_merged['Close'] > 0).all() if 'Close' in df_merged.columns else "N/A",
                'volume_non_negative': (df_merged['Volume'] >= 0).all() if 'Volume' in df_merged.columns else "N/A",
            }
        }
        return checks


# ==============================================================================
# 🎯 ENHANCED HIDDEN GEM ANALYZER
# ==============================================================================
class EnhancedHiddenGemAnalyzer:
    """Enhanced analyzer with predictive features and volatility scoring"""
    
    def __init__(self, df_merged):
        self.df = df_merged.copy()
        self.latest_date = self.df['Date'].max()
        self.error_handler = ErrorHandler()
        self.predictive = PredictiveAnalytics(df_merged)
    
    @PerformanceMonitor.time_execution
    def calculate_enhanced_gem_score(self, stock_code, lookback_days=90):
        """Enhanced scoring with volatility and market regime"""
        return self.error_handler.safe_execute(
            self._calculate_enhanced_gem_score_internal,
            stock_code, lookback_days,
            fallback_value={'total_score': 0, 'signal': 'ERROR', 'signal_color': 'neutral'}
        )
    
    def _calculate_enhanced_gem_score_internal(self, stock_code, lookback_days=90):
        """Internal enhanced scoring logic"""
        stock_data = self.df[self.df['Stock Code'] == stock_code].sort_values('Date')
        if stock_data.empty:
            return {'total_score': 0, 'signal': 'NO DATA', 'signal_color': 'neutral'}
        
        cutoff_date = self.latest_date - timedelta(days=lookback_days)
        recent_data = stock_data[stock_data['Date'] >= cutoff_date]
        
        if recent_data.empty or len(recent_data) < 20:
            return {'total_score': 0, 'signal': 'INSUFFICIENT DATA', 'signal_color': 'neutral'}
        
        latest = recent_data.iloc[-1]
        monthly_data = self.get_monthly_ksei_data(recent_data)
        
        scores = {}
        
        # 1. SMART MONEY ACCUMULATION (40%)
        sm_score = self._calculate_smart_money_score(monthly_data)
        scores['smart_money'] = min(100, sm_score)
        
        # 2. TECHNICAL ANALYSIS (30%)
        tech_score = self._calculate_technical_score(recent_data, latest)
        scores['technical'] = min(100, tech_score)
        
        # 3. FUNDAMENTAL & STRUCTURAL (15%)
        funda_score = self._calculate_fundamental_score(recent_data, latest)
        scores['fundamental'] = min(100, funda_score)
        
        # 4. VOLATILITY SCORING (15%) - NEW!
        vol_score = self._calculate_volatility_score(recent_data)
        scores['volatility'] = min(100, vol_score)
        
        # 5. MARKET REGIME ADJUSTMENT - NEW!
        market_trend = self.analyze_market_trend()
        regime_multiplier = Config.REGIME_MULTIPLIERS.get(market_trend, 1.0)
        
        # Calculate weighted total
        total_score = (
            scores['smart_money'] * Config.SCORE_WEIGHTS['smart_money'] +
            scores['technical'] * Config.SCORE_WEIGHTS['technical'] +
            scores['fundamental'] * Config.SCORE_WEIGHTS['fundamental'] +
            scores['volatility'] * Config.SCORE_WEIGHTS['volatility']
        ) * regime_multiplier
        
        # Apply confidence adjustment based on data quality
        data_quality = self._assess_data_quality(stock_data)
        confidence_multiplier = 0.8 + (data_quality * 0.2)  # 0.8 to 1.0
        
        final_score = min(100, max(0, total_score * confidence_multiplier))
        
        # Get predictive forecast
        forecast = self.predictive.forecast_next_month_flow(stock_code)
        regime_change = self.predictive.detect_regime_change(stock_code)
        
        # Determine signal
        signal, signal_color = self._determine_signal(final_score, forecast, regime_change)
        
        return {
            'total_score': round(final_score, 1),
            'component_scores': scores,
            'signal': signal,
            'signal_color': signal_color,
            'latest_price': latest.get('Close', latest.get('Price', 0)),
            'price_change_period': self._calculate_price_change(recent_data),
            'sector': latest.get('Sector', 'N/A'),
            'free_float': latest.get('Free Float', 0),
            'smart_money_total': self._get_smart_money_total(monthly_data),
            'positive_months': self._count_positive_months(monthly_data),
            'total_months': len(monthly_data) if not monthly_data.empty else 0,
            'monthly_data_points': len(monthly_data),
            'volume_ratio': self._calculate_volume_ratio(recent_data),
            'rsi': latest.get('RSI_14', 50),
            'volatility': self._calculate_annualized_volatility(recent_data),
            'forecast': forecast,
            'regime_change': regime_change,
            'data_quality_score': data_quality
        }
    
    def _calculate_smart_money_score(self, monthly_data):
        """Calculate smart money accumulation score"""
        if monthly_data.empty or 'Smart_Money_Flow' not in monthly_data.columns:
            return 50
        
        sm_total = monthly_data['Smart_Money_Flow'].sum()
        positive_months = (monthly_data['Smart_Money_Flow'] > 0).sum()
        total_months = len(monthly_data)
        
        # Amount score (0-30 points)
        amount_score = min(30, (abs(sm_total) / 5e9) * 15) if abs(sm_total) > 0 else 0
        
        # Consistency score (0-25 points)
        consistency_score = (positive_months / max(total_months, 1)) * 25
        
        # Trend score (0-20 points)
        if len(monthly_data) >= 2:
            sm_values = monthly_data['Smart_Money_Flow'].values
            if len(sm_values) >= 3:
                trend = np.polyfit(range(len(sm_values)), sm_values, 1)[0]
                trend_score = min(20, max(0, trend / 1e9 * 8))
            else:
                trend_score = 12 if sm_values[-1] > sm_values[0] else 0
        else:
            trend_score = 0
        
        # Retail divergence (0-15 points)
        if 'Retail_Flow' in monthly_data.columns:
            retail_total = monthly_data['Retail_Flow'].sum()
            if sm_total > 0 and retail_total < 0:
                divergence_score = 15
            elif sm_total > retail_total * 2:
                divergence_score = 10
            else:
                divergence_score = 0
        else:
            divergence_score = 0
        
        # Concentration score (0-10 points)
        if 'Ownership_Concentration' in monthly_data.columns:
            avg_concentration = monthly_data['Ownership_Concentration'].mean()
            if avg_concentration > 60:
                concentration_score = 10
            elif avg_concentration > 40:
                concentration_score = 7
            else:
                concentration_score = 3
        else:
            concentration_score = 5
        
        return amount_score + consistency_score + trend_score + divergence_score + concentration_score
    
    def _calculate_technical_score(self, recent_data, latest):
        """Calculate technical analysis score"""
        tech_score = 50
        
        if 'Close' in recent_data.columns:
            price_change = self._calculate_price_change(recent_data)
            
            # Price position score (0-20)
            if -15 <= price_change <= 25:
                price_score = 20
            elif price_change < -15:
                price_score = max(5, 25 + price_change)
            else:
                price_score = max(0, 25 - price_change)
            
            # Volume trend (0-15)
            volume_score = self._calculate_volume_score(recent_data)
            
            # RSI score (0-15)
            rsi_score = self._calculate_rsi_score(latest)
            
            # Moving average alignment (0-10)
            ma_score = self._calculate_ma_score(latest)
            
            # Breakout detection (0-10) - NEW!
            breakout_score = self._detect_breakout(recent_data)
            
            tech_score = price_score + volume_score + rsi_score + ma_score + breakout_score
        
        return min(100, max(0, tech_score))
    
    def _calculate_fundamental_score(self, recent_data, latest):
        """Calculate fundamental score"""
        funda_score = 50
        
        # Free Float analysis (0-20)
        ff_score = self._calculate_free_float_score(latest)
        
        # Liquidity score (0-15)
        liquidity_score = self._calculate_liquidity_score(recent_data, latest)
        
        # Sector momentum (0-10)
        sector_score = self._calculate_sector_momentum_score(latest.get('Sector', 'N/A'))
        
        # Market cap position (0-5) - NEW!
        mcap_score = self._calculate_market_cap_score(latest)
        
        funda_score = ff_score + liquidity_score + sector_score + mcap_score
        return min(100, max(0, funda_score))
    
    def _calculate_volatility_score(self, recent_data):
        """Calculate volatility score (lower volatility = higher score)"""
        if 'Close' not in recent_data.columns or len(recent_data) < 20:
            return 50
        
        returns = recent_data['Close'].pct_change().dropna()
        if len(returns) < 10:
            return 50
        
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        # Lower volatility = better for hidden gems (0-100 points)
        if volatility < 0.25:  # < 25% annualized volatility
            return 100
        elif volatility < 0.35:
            return 85
        elif volatility < 0.50:
            return 70
        elif volatility < 0.75:
            return 50
        elif volatility < 1.00:
            return 30
        else:
            return 15
    
    def _calculate_price_change(self, recent_data):
        """Calculate price change over period"""
        if len(recent_data) < 2 or 'Close' not in recent_data.columns:
            return 0
        return ((recent_data.iloc[-1]['Close'] - recent_data.iloc[0]['Close']) / 
                recent_data.iloc[0]['Close'] * 100)
    
    def _calculate_volume_score(self, recent_data):
        """Calculate volume trend score"""
        if 'Volume' not in recent_data.columns or len(recent_data) < 10:
            return 10
        
        avg_volume = recent_data['Volume'].mean()
        recent_avg = recent_data['Volume'].tail(10).mean()
        
        if avg_volume > 0:
            volume_ratio = recent_avg / avg_volume
            if volume_ratio > 1.5:
                return 15
            elif volume_ratio > 1.2:
                return 12
            elif volume_ratio > 0.8:
                return 8
            else:
                return 5
        return 5
    
    def _calculate_rsi_score(self, latest):
        """Calculate RSI score"""
        if 'RSI_14' not in latest:
            return 8
        
        rsi = latest['RSI_14']
        if 30 <= rsi <= 40:  # Mildly oversold = best for accumulation
            return 15
        elif rsi < 30:  # Very oversold
            return 20
        elif 40 < rsi < 60:  # Neutral
            return 10
        elif 60 <= rsi <= 70:  # Mildly overbought
            return 5
        else:  # rsi > 70:  # Overbought
            return 0
    
    def _calculate_ma_score(self, latest):
        """Calculate moving average score"""
        if 'Price_MA20' not in latest or 'Price_MA50' not in latest:
            return 5
        
        if latest['Close'] > latest['Price_MA20'] > latest['Price_MA50']:
            return 10
        elif latest['Close'] > latest['Price_MA20']:
            return 7
        else:
            return 3
    
    def _detect_breakout(self, recent_data):
        """Detect price breakout patterns"""
        if len(recent_data) < 20 or 'Close' not in recent_data.columns:
            return 5
        
        recent_prices = recent_data['Close'].tail(10).values
        prev_prices = recent_data['Close'].iloc[-20:-10].values
        
        if len(recent_prices) < 5 or len(prev_prices) < 5:
            return 5
        
        recent_high = np.max(recent_prices)
        prev_high = np.max(prev_prices)
        
        if recent_high > prev_high * 1.05:  # 5% breakout
            return 10
        elif recent_high > prev_high * 1.02:  # 2% breakout
            return 7
        else:
            return 5
    
    def _calculate_free_float_score(self, latest):
        """Calculate free float score"""
        if 'Free Float' not in latest:
            return 10
        
        ff = latest['Free Float']
        if 20 <= ff <= 40:  # Ideal range for hidden gems
            return 20
        elif ff < 20:  # Too concentrated
            return 15
        elif ff < 10:  # Very concentrated
            return 10
        elif ff > 60:  # Too liquid, less room for institutional accumulation
            return 5
        else:  # 40-60%
            return 15
    
    def _calculate_liquidity_score(self, recent_data, latest):
        """Calculate liquidity score"""
        if 'Value' not in recent_data.columns:
            return 8
        
        avg_daily_value = recent_data['Value'].mean()
        
        if 'Listed Shares' in latest and latest['Close'] > 0:
            market_cap = latest['Close'] * latest['Listed Shares']
            liquidity_ratio = avg_daily_value / market_cap * 100 if market_cap > 0 else 0
        else:
            liquidity_ratio = 0
        
        if liquidity_ratio > 1.0:  # >1% daily turnover
            return 15
        elif liquidity_ratio > 0.5:  # >0.5% daily turnover
            return 12
        elif liquidity_ratio > 0.2:  # >0.2% daily turnover
            return 10
        elif avg_daily_value > 50e9:  # High absolute value
            return 12
        elif avg_daily_value > 20e9:
            return 10
        else:
            return 5
    
    def _calculate_sector_momentum_score(self, sector):
        """Calculate sector momentum score (placeholder)"""
        return 8  # Placeholder - could be enhanced with sector rotation data
    
    def _calculate_market_cap_score(self, latest):
        """Calculate market cap score (smaller = better for hidden gems)"""
        if 'Close' not in latest or 'Listed Shares' not in latest:
            return 3
        
        market_cap = latest['Close'] * latest['Listed Shares']
        market_cap_t = market_cap / 1e12  # Convert to trillions
        
        if market_cap_t < 1:  # Small cap < 1T
            return 5
        elif market_cap_t < 5:  # Mid cap 1-5T
            return 3
        elif market_cap_t < 20:  # Large cap 5-20T
            return 1
        else:  # Mega cap > 20T
            return 0
    
    def _calculate_annualized_volatility(self, recent_data):
        """Calculate annualized volatility"""
        if 'Close' not in recent_data.columns or len(recent_data) < 20:
            return 0
        
        returns = recent_data['Close'].pct_change().dropna()
        if len(returns) < 10:
            return 0
        
        return returns.std() * np.sqrt(252)
    
    def _assess_data_quality(self, stock_data):
        """Assess data quality (0.0 to 1.0)"""
        if stock_data.empty:
            return 0.0
        
        quality_score = 0.0
        
        # Check KSEI data coverage
        if 'Smart_Money_Flow' in stock_data.columns:
            ksei_coverage = stock_data['Smart_Money_Flow'].notna().sum() / len(stock_data)
            quality_score += ksei_coverage * 0.4
        
        # Check trading data coverage
        if 'Close' in stock_data.columns:
            trading_coverage = stock_data['Close'].notna().sum() / len(stock_data)
            quality_score += trading_coverage * 0.4
        
        # Check data recency
        days_since_last = (self.latest_date - stock_data['Date'].max()).days
        recency_score = max(0, 1 - (days_since_last / 30))  # Penalize if >30 days old
        quality_score += recency_score * 0.2
        
        return min(1.0, quality_score)
    
    def _determine_signal(self, score, forecast, regime_change):
        """Determine buy/sell signal"""
        if score >= 85:
            if regime_change == "ACCELERATING_ACCUMULATION":
                return "💎 STRONG BUY - ACCELERATING", "buy"
            return "💎 STRONG BUY", "buy"
        elif score >= 75:
            if forecast and forecast.get('trend') == 'up':
                return "🔥 ACCUMULATE - POSITIVE FORECAST", "buy"
            return "🔥 ACCUMULATE", "buy"
        elif score >= 65:
            if regime_change == "ACCELERATING_DISTRIBUTION":
                return "⚠️ CAUTION - DISTRIBUTING", "neutral"
            return "📈 WATCH", "neutral"
        elif score >= 55:
            return "⚖️ NEUTRAL", "neutral"
        else:
            if regime_change == "ACCELERATING_DISTRIBUTION":
                return "🚫 AVOID - HEAVY DISTRIBUTION", "sell"
            return "⏸️ AVOID", "sell"
    
    def get_monthly_ksei_data(self, stock_data):
        """Extract monthly KSEI data points"""
        if stock_data.empty:
            return pd.DataFrame()
        
        stock_data = stock_data.copy()
        stock_data['Month'] = stock_data['Date'].dt.to_period('M')
        
        monthly_data = stock_data.dropna(subset=['Smart_Money_Flow']).copy()
        if not monthly_data.empty:
            monthly_data = monthly_data.sort_values('Date').groupby('Month').last().reset_index()
        
        return monthly_data
    
    def _get_smart_money_total(self, monthly_data):
        """Get total smart money flow"""
        if monthly_data.empty or 'Smart_Money_Flow' not in monthly_data.columns:
            return 0
        return monthly_data['Smart_Money_Flow'].sum()
    
    def _count_positive_months(self, monthly_data):
        """Count positive smart money months"""
        if monthly_data.empty or 'Smart_Money_Flow' not in monthly_data.columns:
            return 0
        return (monthly_data['Smart_Money_Flow'] > 0).sum()
    
    def _calculate_volume_ratio(self, recent_data):
        """Calculate volume ratio"""
        if 'Volume' not in recent_data.columns or len(recent_data) < 10:
            return 1.0
        
        avg_volume = recent_data['Volume'].mean()
        recent_avg = recent_data['Volume'].tail(10).mean()
        
        if avg_volume > 0:
            return recent_avg / avg_volume
        return 1.0
    
    def analyze_market_trend(self):
        """Analyze overall market trend"""
        # Simple market trend analysis based on index stocks
        index_stocks = ['BBCA', 'BBRI', 'BMRI', 'TLKM', 'ASII']
        index_data = self.df[self.df['Stock Code'].isin(index_stocks)]
        
        if index_data.empty:
            return 'neutral'
        
        latest_index = index_data[index_data['Date'] == self.latest_date]
        if latest_index.empty:
            return 'neutral'
        
        avg_change = latest_index['Change %'].mean() if 'Change %' in latest_index.columns else 0
        
        if avg_change > 1.0:
            return 'bull'
        elif avg_change < -1.0:
            return 'bear'
        elif avg_change < -0.5:
            return 'correction'
        else:
            return 'neutral'
    
    @PerformanceMonitor.time_execution
    def find_top_gems(self, top_n=25, min_score=65, sector_filter=None):
        """Find top hidden gem candidates with parallel processing"""
        unique_stocks = self.df['Stock Code'].unique()[:Config.MAX_STOCKS_ANALYZED]
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_stocks = len(unique_stocks)
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
            futures = {}
            
            for i, stock in enumerate(unique_stocks):
                futures[executor.submit(self.calculate_enhanced_gem_score, stock)] = stock
            
            completed = 0
            for future in as_completed(futures):
                completed += 1
                stock = futures[future]
                
                try:
                    score_data = future.result(timeout=5)
                    
                    if score_data and score_data['total_score'] >= min_score:
                        if sector_filter and score_data['sector'] != sector_filter:
                            continue
                        
                        results.append({
                            'Stock': stock,
                            'Gem Score': score_data['total_score'],
                            'Signal': score_data['signal'],
                            'Sector': score_data['sector'],
                            'Price': score_data['latest_price'],
                            'Price Chg': score_data['price_change_period'],
                            'Free Float %': score_data['free_float'],
                            'Smart Money (B)': score_data['smart_money_total'] / 1e9,
                            'Positive Months': score_data['positive_months'],
                            'Total Months': score_data['total_months'],
                            'Monthly Data': score_data['monthly_data_points'],
                            'RSI': score_data['rsi'],
                            'Volatility %': score_data.get('volatility', 0) * 100,
                            'Volume Trend': score_data.get('volume_ratio', 1),
                            'Smart Score': score_data['component_scores'].get('smart_money', 0),
                            'Tech Score': score_data['component_scores'].get('technical', 0),
                            'Fundamental Score': score_data['component_scores'].get('fundamental', 0),
                            'Volatility Score': score_data['component_scores'].get('volatility', 0),
                            'Data Quality': score_data.get('data_quality_score', 0) * 100,
                            'Regime Change': score_data.get('regime_change', 'STABLE')
                        })
                
                except Exception as e:
                    self.error_handler.logger.error(f"Error processing {stock}: {e}")
                
                # Update progress
                progress = completed / total_stocks
                progress_bar.progress(progress)
                status_text.text(f"🔍 Analyzing... {completed}/{total_stocks} stocks")
        
        progress_bar.empty()
        status_text.empty()
        
        if results:
            df_results = pd.DataFrame(results)
            df_results = df_results.sort_values('Gem Score', ascending=False).head(top_n)
            return df_results
        else:
            return pd.DataFrame()


# ==============================================================================
# 🔮 PREDICTIVE ANALYTICS MODULE
# ==============================================================================
class PredictiveAnalytics:
    """Predictive analytics for forecasting and regime detection"""
    
    def __init__(self, df_merged):
        self.df = df_merged
    
    def forecast_next_month_flow(self, stock_code):
        """ARIMA-like forecasting for next month smart money flow"""
        stock_data = self.df[self.df['Stock Code'] == stock_code].sort_values('Date')
        
        if 'Smart_Money_Flow' not in stock_data.columns or len(stock_data) < 6:
            return None
        
        # Get month-end data
        monthly = stock_data[stock_data['Date'].dt.is_month_end]
        if len(monthly) < 3:
            return None
        
        sm_flow = monthly['Smart_Money_Flow'].dropna().values
        
        # Simple exponential smoothing forecast
        alpha = 0.3  # Smoothing factor
        forecast = sm_flow[-1]
        for i in range(len(sm_flow) - 2, -1, -1):
            forecast = alpha * sm_flow[i] + (1 - alpha) * forecast
        
        # Calculate confidence based on data consistency
        volatility = np.std(sm_flow) / max(np.abs(np.mean(sm_flow)), 1e9)
        confidence = max(0.1, 0.9 - volatility)
        
        trend = 'up' if forecast > sm_flow[-1] else 'down'
        magnitude = abs(forecast - sm_flow[-1]) / max(abs(sm_flow[-1]), 1e9)
        
        return {
            'forecast': forecast,
            'confidence': confidence,
            'trend': trend,
            'magnitude_change': magnitude,
            'direction': 'BUY' if forecast > 0 else 'SELL',
            'strength': 'STRONG' if magnitude > 0.5 else 'MODERATE' if magnitude > 0.2 else 'WEAK'
        }
    
    def detect_regime_change(self, stock_code, window=3):
        """Detect regime changes in ownership pattern"""
        stock_data = self.df[self.df['Stock Code'] == stock_code].sort_values('Date')
        
        if 'Smart_Money_Flow' not in stock_data.columns or len(stock_data) < window * 2:
            return None
        
        monthly = stock_data[stock_data['Date'].dt.is_month_end]
        if len(monthly) < window * 2:
            return None
        
        recent = monthly['Smart_Money_Flow'].tail(window).values
        previous = monthly['Smart_Money_Flow'].iloc[-window*2:-window].values
        
        if len(recent) < window or len(previous) < window:
            return None
        
        recent_mean = np.mean(recent)
        previous_mean = np.mean(previous)
        
        # Calculate change significance
        if abs(previous_mean) > 0:
            change_ratio = recent_mean / previous_mean
        else:
            change_ratio = float('inf') if recent_mean != 0 else 1.0
        
        # Detect regime
        if recent_mean > previous_mean * 1.5:
            return "ACCELERATING_ACCUMULATION"
        elif recent_mean < previous_mean * 0.5:
            return "ACCELERATING_DISTRIBUTION"
        elif abs(recent_mean) > abs(previous_mean) * 1.2:
            if recent_mean > 0:
                return "INTENSIFYING_ACCUMULATION"
            else:
                return "INTENSIFYING_DISTRIBUTION"
        elif abs(recent_mean) < abs(previous_mean) * 0.8:
            return "DECELERATING"
        else:
            return "STABLE"
    
    def calculate_correlation_matrix(self, gem_data):
        """Calculate correlation between gem factors"""
        if gem_data.empty or len(gem_data) < 5:
            return None
        
        numeric_cols = [
            'Gem Score', 'Smart Money (B)', 'Free Float %', 
            'RSI', 'Volatility %', 'Price Chg', 'Data Quality'
        ]
        numeric_cols = [col for col in numeric_cols if col in gem_data.columns]
        
        correlation_data = gem_data[numeric_cols].corr()
        return correlation_data


# ==============================================================================
# 📊 ENHANCED VISUALIZATION FUNCTIONS
# ==============================================================================
class EnhancedVisualizations:
    """Enhanced visualization functions with professional styling"""
    
    @staticmethod
    def create_gem_radar_chart(scores, stock_code, signal):
        """Create radar chart for gem score components"""
        categories = ['Smart Money', 'Technical', 'Fundamental', 'Volatility']
        values = [
            scores.get('smart_money', 0),
            scores.get('technical', 0),
            scores.get('fundamental', 0),
            scores.get('volatility', 0)
        ]
        
        fig = go.Figure(data=go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor='rgba(255, 215, 0, 0.2)',
            line=dict(color='#FFD700', width=3),
            marker=dict(size=8, color='#4318FF')
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True, 
                    range=[0, 100],
                    gridcolor='rgba(67, 24, 255, 0.1)',
                    linecolor='rgba(67, 24, 255, 0.3)'
                ),
                angularaxis=dict(
                    gridcolor='rgba(67, 24, 255, 0.1)',
                    linecolor='rgba(67, 24, 255, 0.3)'
                ),
                bgcolor='rgba(0,0,0,0)'
            ),
            showlegend=False,
            title=dict(
                text=f"💎 {stock_code} - {signal}",
                font=dict(size=18, color='#2B3674')
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#2B3674'),
            height=400,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        return fig
    
    @staticmethod
    def create_ownership_timeline(df, stock_code):
        """Create enhanced timeline of ownership changes"""
        stock_data = df[df['Stock Code'] == stock_code].sort_values('Date')
        
        if stock_data.empty:
            return None
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                f'{stock_code} - Price History & Moving Averages',
                'Smart Money Flow (B) - Monthly',
                'Institutional Net Flow (B)'
            ),
            vertical_spacing=0.08,
            row_heights=[0.5, 0.25, 0.25],
            shared_xaxes=True
        )
        
        # Row 1: Price with MAs
        fig.add_trace(
            go.Scatter(
                x=stock_data['Date'], 
                y=stock_data['Close'], 
                name='Price', 
                line=dict(color='#4318FF', width=3),
                mode='lines'
            ),
            row=1, col=1
        )
        
        if 'Price_MA20' in stock_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=stock_data['Date'], 
                    y=stock_data['Price_MA20'], 
                    name='MA20', 
                    line=dict(color='#FFB547', width=2, dash='dash'),
                    mode='lines'
                ),
                row=1, col=1
            )
        
        if 'Price_MA50' in stock_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=stock_data['Date'], 
                    y=stock_data['Price_MA50'], 
                    name='MA50', 
                    line=dict(color='#05CD99', width=2, dash='dash'),
                    mode='lines'
                ),
                row=1, col=1
            )
        
        # Row 2: Smart Money Flow
        month_ends = stock_data[stock_data['Date'].dt.is_month_end]
        if not month_ends.empty and 'Smart_Money_Flow' in month_ends.columns:
            colors = ['#05CD99' if x > 0 else '#EE5D50' for x in month_ends['Smart_Money_Flow']]
            fig.add_trace(
                go.Bar(
                    x=month_ends['Date'], 
                    y=month_ends['Smart_Money_Flow'] / 1e9, 
                    name='Smart Money (B)', 
                    marker_color=colors,
                    opacity=0.8
                ),
                row=2, col=1
            )
        
        # Row 3: Institutional Net Flow
        if not month_ends.empty and 'Institutional_Net' in month_ends.columns:
            fig.add_trace(
                go.Bar(
                    x=month_ends['Date'], 
                    y=month_ends['Institutional_Net'] / 1e9, 
                    name='Institutional Net (B)', 
                    marker_color='#4318FF',
                    opacity=0.6
                ),
                row=3, col=1
            )
        
        fig.update_layout(
            height=700,
            showlegend=True,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#2B3674'),
            hovermode='x unified',
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=1.02, 
                xanchor="right", 
                x=1
            ),
            margin=dict(l=10, r=10, t=40, b=10)
        )
        
        fig.update_xaxes(showgrid=True, gridcolor='#E0E5F2')
        fig.update_yaxes(showgrid=True, gridcolor='#E0E5F2')
        
        return fig
    
    @staticmethod
    def create_sector_heatmap(gems_df):
        """Create heatmap of gems by sector"""
        if gems_df.empty or 'Sector' not in gems_df.columns:
            return None
        
        sector_stats = gems_df.groupby('Sector').agg({
            'Stock': 'count',
            'Gem Score': 'mean',
            'Smart Money (B)': 'sum',
            'Volatility %': 'mean'
        }).reset_index()
        
        sector_stats.columns = ['Sector', 'Count', 'Avg Score', 'Total Smart Money (B)', 'Avg Volatility']
        
        fig = px.treemap(
            sector_stats,
            path=['Sector'],
            values='Total Smart Money (B)',
            color='Avg Score',
            color_continuous_scale='RdYlGn',
            title='💎 Hidden Gems by Sector',
            hover_data=['Count', 'Avg Score', 'Avg Volatility'],
            labels={'color': 'Avg Score'}
        )
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#2B3674'),
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        return fig
    
    @staticmethod
    def create_correlation_heatmap(correlation_data):
        """Create correlation heatmap for gem factors"""
        if correlation_data is None:
            return None
        
        fig = px.imshow(
            correlation_data,
            text_auto='.2f',
            color_continuous_scale='RdBu',
            title="💎 Factor Correlation Heatmap",
            aspect="auto",
            labels=dict(color="Correlation")
        )
        
        fig.update_layout(
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#2B3674'),
            xaxis_title="Factors",
            yaxis_title="Factors"
        )
        
        return fig
    
    @staticmethod
    def create_gem_timeline_evolution(analyzer, stock_code, df_merged):
        """Show how gem score evolved over time"""
        dates = df_merged[df_merged['Stock Code'] == stock_code]['Date'].unique()
        scores = []
        
        for date in sorted(dates)[-12:]:  # Last 12 data points
            temp_df = df_merged[df_merged['Date'] <= date]
            temp_analyzer = EnhancedHiddenGemAnalyzer(temp_df)
            score_data = temp_analyzer.calculate_enhanced_gem_score(stock_code, 90)
            
            if score_data:
                scores.append({
                    'Date': date,
                    'Score': score_data['total_score'],
                    'Smart_Money': score_data['smart_money_total'] / 1e9,
                    'Signal': score_data['signal']
                })
        
        if not scores:
            return None
        
        df_scores = pd.DataFrame(scores)
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Score line
        fig.add_trace(
            go.Scatter(
                x=df_scores['Date'],
                y=df_scores['Score'],
                name='Gem Score',
                line=dict(color='#FFD700', width=3),
                mode='lines+markers'
            ),
            secondary_y=False
        )
        
        # Smart Money bars
        fig.add_trace(
            go.Bar(
                x=df_scores['Date'],
                y=df_scores['Smart_Money'],
                name='Smart Money (B)',
                marker_color='#4318FF',
                opacity=0.6
            ),
            secondary_y=True
        )
        
        fig.update_layout(
            title=dict(
                text=f"📈 {stock_code} - Score Evolution",
                font=dict(size=18, color='#2B3674')
            ),
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#2B3674'),
            hovermode='x unified',
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=1.02, 
                xanchor="right", 
                x=1
            )
        )
        
        fig.update_yaxes(title_text="Gem Score", secondary_y=False)
        fig.update_yaxes(title_text="Smart Money (B)", secondary_y=True)
        
        return fig


# ==============================================================================
# 💼 PORTFOLIO SIMULATOR & REPORT GENERATOR
# ==============================================================================
class PortfolioSimulator:
    """Portfolio simulation and reporting"""
    
    @staticmethod
    def simulate_portfolio(gems_df, capital, top_n=10):
        """Simulate portfolio with top N gems"""
        if gems_df.empty or len(gems_df) < top_n:
            return None
        
        top_gems = gems_df.head(top_n).copy()
        allocation_per_stock = capital / len(top_gems)
        
        portfolio = []
        for idx, row in top_gems.iterrows():
            portfolio.append({
                'Stock': row['Stock'],
                'Sector': row['Sector'],
                'Allocation': allocation_per_stock,
                'Gem Score': row['Gem Score'],
                'Signal': row['Signal'],
                'Risk Level': 'LOW' if row['Volatility %'] < 30 else 'MEDIUM' if row['Volatility %'] < 50 else 'HIGH',
                'Expected Return': min(30, row['Gem Score'] * 0.3)  # Simple heuristic
            })
        
        df_portfolio = pd.DataFrame(portfolio)
        
        # Calculate portfolio metrics
        total_expected_return = df_portfolio['Expected Return'].mean()
        weighted_score = (df_portfolio['Gem Score'] * df_portfolio['Allocation']).sum() / capital
        
        sector_allocation = df_portfolio.groupby('Sector')['Allocation'].sum().reset_index()
        sector_allocation['Percentage'] = (sector_allocation['Allocation'] / capital) * 100
        
        return {
            'portfolio': df_portfolio,
            'metrics': {
                'total_capital': capital,
                'number_of_stocks': len(top_gems),
                'average_gem_score': df_portfolio['Gem Score'].mean(),
                'weighted_score': weighted_score,
                'expected_return': total_expected_return,
                'sector_diversification': len(sector_allocation)
            },
            'sector_allocation': sector_allocation
        }
    
    @staticmethod
    def generate_report(gems_df, analysis_date, file_format='csv'):
        """Generate comprehensive report"""
        from io import BytesIO
        
        if gems_df.empty:
            return None
        
        if file_format == 'csv':
            output = BytesIO()
            gems_df.to_csv(output, index=False)
            output.seek(0)
            return output.getvalue()
        
        elif file_format == 'excel':
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Summary sheet
                summary = pd.DataFrame({
                    'Report Date': [analysis_date.strftime('%Y-%m-%d')],
                    'Total Gems': [len(gems_df)],
                    'Average Gem Score': [gems_df['Gem Score'].mean()],
                    'Total Smart Money (B)': [gems_df['Smart Money (B)'].sum()],
                    'Top Sector': [gems_df['Sector'].mode().iloc[0] if not gems_df.empty else 'N/A']
                })
                summary.to_excel(writer, sheet_name='Summary', index=False)
                
                # Detailed analysis
                gems_df.to_excel(writer, sheet_name='Detailed Analysis', index=False)
                
                # Sector analysis
                sector_stats = gems_df.groupby('Sector').agg({
                    'Stock': 'count',
                    'Gem Score': 'mean',
                    'Smart Money (B)': 'sum',
                    'Volatility %': 'mean'
                }).reset_index()
                sector_stats.to_excel(writer, sheet_name='Sector Analysis', index=False)
                
                # Risk analysis
                risk_stats = pd.DataFrame({
                    'Risk Level': ['Low (Vol < 30%)', 'Medium (30-50%)', 'High (>50%)'],
                    'Count': [
                        len(gems_df[gems_df['Volatility %'] < 30]),
                        len(gems_df[(gems_df['Volatility %'] >= 30) & (gems_df['Volatility %'] < 50)]),
                        len(gems_df[gems_df['Volatility %'] >= 50])
                    ]
                })
                risk_stats.to_excel(writer, sheet_name='Risk Analysis', index=False)
            
            output.seek(0)
            return output.getvalue()


# ==============================================================================
# 🚀 MAIN DASHBOARD - ENTERPRISE EDITION
# ==============================================================================
def main():
    """Main dashboard function - Enterprise Edition"""
    
    # HEADER
    st.markdown("""
    <div class="header-gradient">
        <div class="header-title">🚀 HIDDEN GEM FINDER v3.0</div>
        <div class="header-subtitle">Enterprise Edition • Predictive Analytics • Portfolio Simulation • Real-time Monitoring</div>
        <div style="margin-top: 20px; display: flex; gap: 10px; flex-wrap: wrap;">
            <div style="background: rgba(255,255,255,0.2); padding: 6px 16px; border-radius: 20px;">📈 Monthly KSEI Intelligence</div>
            <div style="background: rgba(255,255,255,0.2); padding: 6px 16px; border-radius: 20px;">🔮 Predictive Analytics</div>
            <div style="background: rgba(255,255,255,0.2); padding: 6px 16px; border-radius: 20px;">💼 Portfolio Simulation</div>
            <div style="background: rgba(255,255,255,0.2); padding: 6px 16px; border-radius: 20px;">📊 Real-time Monitoring</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize data loader
    loader = EnhancedDataLoader()
    
    if not loader.service:
        st.error("❌ Failed to initialize Google Drive service")
        st.stop()
    
    # Load data with enhanced monitoring
    with st.spinner("🚀 Loading Enhanced Data Pipeline..."):
        df_merged = loader.load_merged_data()
    
    if df_merged.empty:
        st.error("❌ Failed to load data. Please check data files and credentials.")
        st.stop()
    
    # Data validation
    data_checks = loader.validate_data_integrity(df_merged)
    if data_checks['null_percentage'] > 30:
        st.warning(f"⚠️ High missing data: {data_checks['null_percentage']}. Consider data quality.")
    
    # Store in session state
    analyzer = EnhancedHiddenGemAnalyzer(df_merged)
    st.session_state.analyzer = analyzer
    st.session_state.df_merged = df_merged
    st.session_state.data_checks = data_checks
    
    # Initialize visualizations
    viz = EnhancedVisualizations()
    
    # SIDEBAR - ENHANCED
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=70)
        st.markdown("<h2 style='color:#2B3674; margin-top: 0;'>💎 Gem Finder v3.0</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color:#A3AED0; font-size: 14px;'>Enterprise Analytics Platform</p>", unsafe_allow_html=True)
        st.divider()
        
        # Data Overview
        st.markdown("##### 📊 Data Overview")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric("Stocks", df_merged['Stock Code'].nunique())
        with col_s2:
            st.metric("Latest", df_merged['Date'].max().strftime('%d/%m'))
        
        st.divider()
        
        # Real-time Market Stats
        st.markdown("##### 📈 Real-time Market")
        latest_date = df_merged['Date'].max()
        daily_stats = df_merged[df_merged['Date'] == latest_date]
        
        if not daily_stats.empty:
            col_rt1, col_rt2 = st.columns(2)
            with col_rt1:
                adv = (daily_stats['Change %'] > 0).sum()
                st.metric("Advancers", adv)
            with col_rt2:
                dec = (daily_stats['Change %'] < 0).sum()
                st.metric("Decliners", dec)
            
            avg_change = daily_stats['Change %'].mean()
            st.metric("Avg Change", f"{avg_change:.2f}%")
        
        st.divider()
        
        # Analysis Settings
        st.markdown("##### ⚙️ Analysis Settings")
        lookback_days = st.slider("Analysis Period (Days)", 30, 180, 90, 15)
        min_gem_score = st.slider("Minimum Score", 50, 90, 70, 5)
        top_n_gems = st.slider("Top N Results", 10, 50, 25, 5)
        
        # Sector filter
        raw_sectors = df_merged['Sector'].dropna().unique()
        clean_sectors = [str(x) for x in raw_sectors if str(x).lower() != 'nan']
        sectors = ['All'] + sorted(clean_sectors)
        selected_sector = st.selectbox("Sector Filter", sectors)
        
        # Advanced filters
        with st.expander("🔍 Advanced Filters"):
            min_smart_money = st.number_input("Min Smart Money (B)", 0.0, 100.0, 2.0, 0.5)
            max_free_float = st.number_input("Max Free Float %", 0.0, 100.0, 50.0, 5.0)
            min_rsi = st.slider("Min RSI", 0, 100, 30, 5)
            max_rsi = st.slider("Max RSI", 0, 100, 65, 5)
            max_volatility = st.slider("Max Volatility %", 0, 200, 60, 5)
        
        # Portfolio Simulator
        st.divider()
        st.markdown("##### 💼 Portfolio Simulator")
        sim_capital = st.number_input(
            "Simulation Capital (Rp M)", 
            min_value=100, 
            max_value=10000, 
            value=1000
        ) * 1e6
        
        simulator = PortfolioSimulator()
        
        # Actions
        st.divider()
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🔄 Refresh Data", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        
        with col_btn2:
            if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
                st.session_state.run_analysis = True
        
        # Performance metrics
        PerformanceMonitor.display_performance_metrics()
        
        # Data quality info
        with st.expander("📁 Dataset Quality"):
            st.write(f"**Total Rows:** {data_checks['total_rows']:,}")
            st.write(f"**Date Range:** {data_checks['date_range']}")
            st.write(f"**KSEI Coverage:** {data_checks['ksei_coverage']}")
            st.write(f"**Null Data:** {data_checks['null_percentage']}")
            st.write(f"**Duplicates:** {data_checks['duplicates']}")
    
    # MAIN CONTENT - ENHANCED TABS
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🏆 Top Gems", "📈 Stock Analyzer", "📊 Market Intelligence", 
        "🔄 Sector Rotation", "🔮 Predictive Analytics", "💼 Portfolio Lab", 
        "📁 Data Diagnostics"
    ])
    
    # TAB 1: TOP GEMS - ENHANCED
    with tab1:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">💎 Top Hidden Gem Candidates</div>', unsafe_allow_html=True)
        
        col_t1, col_t2 = st.columns([3, 1])
        with col_t1:
            st.markdown(f"""
            **Analysis Period:** {lookback_days} days | **Min Score:** {min_gem_score} | 
            **Sector:** {selected_sector} | **Real-time:** {df_merged['Date'].max().strftime('%d %b %Y')}
            """)
        
        with col_t2:
            if st.button("🔍 Find Enhanced Gems", type="primary", use_container_width=True):
                st.session_state.find_gems = True
        
        if 'find_gems' in st.session_state and st.session_state.find_gems:
            sector_filter = None if selected_sector == 'All' else selected_sector
            
            with st.spinner(f"🚀 Analyzing {Config.MAX_STOCKS_ANALYZED} stocks with enhanced algorithms..."):
                gems_df = analyzer.find_top_gems(
                    top_n=top_n_gems, 
                    min_score=min_gem_score,
                    sector_filter=sector_filter
                )
            
            if not gems_df.empty:
                # Apply advanced filters
                filtered_df = gems_df.copy()
                filtered_df = filtered_df[filtered_df['Smart Money (B)'] >= min_smart_money]
                filtered_df = filtered_df[filtered_df['Free Float %'] <= max_free_float]
                filtered_df = filtered_df[(filtered_df['RSI'] >= min_rsi) & (filtered_df['RSI'] <= max_rsi)]
                filtered_df = filtered_df[filtered_df['Volatility %'] <= max_volatility]
                
                if not filtered_df.empty:
                    # Summary metrics with enhanced visual
                    col_sm1, col_sm2, col_sm3, col_sm4 = st.columns(4)
                    with col_sm1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div style="font-size:0.9rem; color:#6c757d; margin-bottom:0.5rem;">Total Gems</div>
                            <div style="font-size:2rem; font-weight:700; color:#4318FF;">{len(filtered_df)}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_sm2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div style="font-size:0.9rem; color:#6c757d; margin-bottom:0.5rem;">Avg Score</div>
                            <div style="font-size:2rem; font-weight:700; color:#FFD700;">{filtered_df['Gem Score'].mean():.1f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_sm3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div style="font-size:0.9rem; color:#6c757d; margin-bottom:0.5rem;">Smart Money</div>
                            <div style="font-size:2rem; font-weight:700; color:#05CD99;">{filtered_df['Smart Money (B)'].sum():.1f}B</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_sm4:
                        top_sector = filtered_df['Sector'].mode().iloc[0] if not filtered_df.empty else "N/A"
                        st.markdown(f"""
                        <div class="metric-card">
                            <div style="font-size:0.9rem; color:#6c757d; margin-bottom:0.5rem;">Top Sector</div>
                            <div style="font-size:1.5rem; font-weight:700; color:#868CFF;">{top_sector}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Sector heatmap
                    st.markdown("#### 🗺️ Sector Distribution")
                    heatmap = viz.create_sector_heatmap(filtered_df)
                    if heatmap:
                        st.plotly_chart(heatmap, use_container_width=True)
                    
                    # Correlation heatmap
                    st.markdown("#### 🔗 Factor Correlation")
                    predictive = PredictiveAnalytics(df_merged)
                    correlation_data = predictive.calculate_correlation_matrix(filtered_df)
                    if correlation_data is not None:
                        corr_heatmap = viz.create_correlation_heatmap(correlation_data)
                        if corr_heatmap:
                            st.plotly_chart(corr_heatmap, use_container_width=True)
                    
                    # Gems table with enhanced formatting
                    st.markdown("#### 🏆 Filtered Gem Candidates")
                    
                    display_cols = ['Stock', 'Gem Score', 'Signal', 'Sector', 'Price', 
                                  'Price Chg', 'Free Float %', 'Smart Money (B)', 
                                  'Positive Months', 'Volatility %', 'RSI', 'Data Quality', 'Regime Change']
                    
                    display_df = filtered_df[display_cols].copy()
                    
                    def color_signal(val):
                        if 'BUY' in val:
                            return 'background-color: #D1FAE5; color: #065F46; font-weight: bold;'
                        elif 'ACCUMULATE' in val:
                            return 'background-color: #FEF3C7; color: #92400E; font-weight: bold;'
                        elif 'WATCH' in val:
                            return 'background-color: #DBEAFE; color: #1E40AF; font-weight: bold;'
                        elif 'AVOID' in val or 'CAUTION' in val:
                            return 'background-color: #FEE2E2; color: #991B1B; font-weight: bold;'
                        else:
                            return 'background-color: #F3F4F6; color: #6B7280;'
                    
                    def format_regime(val):
                        if 'ACCELERATING' in val:
                            return f'🚀 {val}'
                        elif 'INTENSIFYING' in val:
                            return f'📈 {val}'
                        elif 'STABLE' in val:
                            return f'⚖️ {val}'
                        else:
                            return val
                    
                    display_df['Regime Change'] = display_df['Regime Change'].apply(format_regime)
                    
                    st.dataframe(
                        display_df.style
                        .format({
                            'Price': 'Rp {:,.0f}',
                            'Price Chg': '{:.1f}%',
                            'Free Float %': '{:.1f}%',
                            'Smart Money (B)': '{:.2f}',
                            'RSI': '{:.1f}',
                            'Gem Score': '{:.1f}',
                            'Volatility %': '{:.1f}%',
                            'Data Quality': '{:.1f}%'
                        })
                        .applymap(color_signal, subset=['Signal'])
                        .background_gradient(subset=['Gem Score'], cmap='RdYlGn', vmin=70, vmax=100)
                        .background_gradient(subset=['Data Quality'], cmap='Blues', vmin=50, vmax=100)
                        .bar(subset=['Smart Money (B)'], color='#05CD99')
                        .bar(subset=['Positive Months'], color='#4318FF'),
                        use_container_width=True,
                        hide_index=True,
                        height=600
                    )
                    
                    # Export options
                    col_exp1, col_exp2, col_exp3 = st.columns(3)
                    with col_exp1:
                        csv = display_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv,
                            file_name=f"hidden_gems_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col_exp2:
                        excel_report = simulator.generate_report(filtered_df, datetime.now(), 'excel')
                        if excel_report:
                            st.download_button(
                                label="📊 Download Excel Report",
                                data=excel_report,
                                file_name=f"gem_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                                mime="application/vnd.ms-excel",
                                use_container_width=True
                            )
                    
                    with col_exp3:
                        if st.button("💼 Simulate Portfolio", use_container_width=True):
                            st.session_state.simulate_portfolio = True
                    
                    # Top gem deep dive
                    if len(filtered_df) > 0:
                        st.markdown("#### 🎯 Top Gem Deep Dive")
                        top_gem = filtered_df.iloc[0]['Stock']
                        score_data = analyzer.calculate_enhanced_gem_score(top_gem, lookback_days)
                        
                        if score_data:
                            col_ana1, col_ana2 = st.columns(2)
                            
                            with col_ana1:
                                st.plotly_chart(
                                    viz.create_gem_radar_chart(
                                        score_data['component_scores'], 
                                        top_gem, 
                                        score_data['signal']
                                    ),
                                    use_container_width=True
                                )
                            
                            with col_ana2:
                                st.plotly_chart(
                                    viz.create_ownership_timeline(df_merged, top_gem),
                                    use_container_width=True
                                )
                            
                            # Timeline evolution
                            st.markdown("##### 📈 Score Evolution")
                            timeline = viz.create_gem_timeline_evolution(analyzer, top_gem, df_merged)
                            if timeline:
                                st.plotly_chart(timeline, use_container_width=True)
                            
                            # Detailed analysis
                            with st.expander(f"📊 Detailed Analysis: {top_gem}"):
                                col_d1, col_d2, col_d3, col_d4 = st.columns(4)
                                with col_d1:
                                    st.metric("Gem Score", f"{score_data['total_score']:.1f}/100")
                                with col_d2:
                                    st.metric("Market Regime", analyzer.analyze_market_trend().upper())
                                with col_d3:
                                    months_pos = f"{score_data['positive_months']}/{score_data['total_months']}"
                                    st.metric("Positive Months", months_pos)
                                with col_d4:
                                    st.metric("Volatility", f"{score_data.get('volatility', 0)*100:.1f}%")
                                
                                # Predictive forecast
                                if score_data.get('forecast'):
                                    forecast = score_data['forecast']
                                    st.markdown("##### 🔮 Predictive Forecast")
                                    col_f1, col_f2, col_f3 = st.columns(3)
                                    with col_f1:
                                        st.metric("Next Month Flow", 
                                                 f"Rp {forecast.get('forecast', 0)/1e9:.1f}B",
                                                 delta=forecast.get('trend', '').upper())
                                    with col_f2:
                                        st.metric("Confidence", f"{forecast.get('confidence', 0)*100:.0f}%")
                                    with col_f3:
                                        st.metric("Strength", forecast.get('strength', 'N/A'))
                else:
                    st.warning("⚠️ No gems found with current filters. Try relaxing criteria.")
            else:
                st.info("📭 No hidden gems found. Try lowering minimum score or changing sector filter.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # TAB 2: STOCK ANALYZER - ENHANCED
    with tab2:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🔍 Individual Stock Analysis</div>', unsafe_allow_html=True)
        
        available_stocks = sorted(df_merged['Stock Code'].unique())
        selected_stock = st.selectbox(
            "Select Stock for Deep Analysis", 
            available_stocks,
            index=available_stocks.index('BBRI') if 'BBRI' in available_stocks else 0,
            key="stock_analyzer_select"
        )
        
        if selected_stock:
            score_data = analyzer.calculate_enhanced_gem_score(selected_stock, lookback_days)
            
            if score_data:
                # Header metrics with enhanced styling
                col_h1, col_h2, col_h3, col_h4 = st.columns(4)
                with col_h1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size:0.8rem; color:#6c757d; margin-bottom:0.5rem;">Gem Score</div>
                        <div style="font-size:2rem; font-weight:700; color:#FFD700;">{score_data['total_score']:.1f}/100</div>
                        <div style="font-size:0.8rem; margin-top:0.5rem; color:#A3AED0;">{score_data['signal']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_h2:
                    price_change = score_data['price_change_period']
                    price_color = "#05CD99" if price_change > 0 else "#EE5D50"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size:0.8rem; color:#6c757d; margin-bottom:0.5rem;">Price & Change</div>
                        <div style="font-size:1.8rem; font-weight:700;">Rp {score_data['latest_price']:,.0f}</div>
                        <div style="font-size:0.9rem; color:{price_color}; font-weight:600;">{price_change:+.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_h3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size:0.8rem; color:#6c757d; margin-bottom:0.5rem;">Smart Money Flow</div>
                        <div style="font-size:1.8rem; font-weight:700; color:#4318FF;">Rp {score_data['smart_money_total']/1e9:.1f}B</div>
                        <div style="font-size:0.8rem; margin-top:0.5rem; color:#A3AED0;">
                            {score_data['positive_months']}/{score_data['total_months']} months positive
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_h4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size:0.8rem; color:#6c757d; margin-bottom:0.5rem;">Fundamental Metrics</div>
                        <div style="font-size:1.2rem; font-weight:700; color:#2B3674;">{score_data['sector']}</div>
                        <div style="display:flex; justify-content:space-between; margin-top:0.5rem;">
                            <span style="font-size:0.8rem; color:#A3AED0;">FF: {score_data['free_float']:.1f}%</span>
                            <span style="font-size:0.8rem; color:#A3AED0;">RSI: {score_data['rsi']:.1f}</span>
                            <span style="font-size:0.8rem; color:#A3AED0;">Vol: {score_data.get('volatility', 0)*100:.1f}%</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Charts
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    st.plotly_chart(
                        viz.create_gem_radar_chart(
                            score_data['component_scores'], 
                            selected_stock, 
                            score_data['signal']
                        ),
                        use_container_width=True
                    )
                
                with col_c2:
                    st.plotly_chart(
                        viz.create_ownership_timeline(df_merged, selected_stock),
                        use_container_width=True
                    )
                
                # Data quality and predictive info
                st.markdown("##### 📊 Advanced Analytics")
                col_q1, col_q2, col_q3, col_q4 = st.columns(4)
                with col_q1:
                    st.metric("Data Quality", f"{score_data.get('data_quality_score', 0)*100:.1f}%")
                with col_q2:
                    pos_rate = (score_data['positive_months'] / max(score_data['total_months'], 1)) * 100
                    st.metric("Positive Rate", f"{pos_rate:.0f}%")
                with col_q3:
                    st.metric("Volume Trend", f"{score_data.get('volume_ratio', 1):.2f}x")
                with col_q4:
                    st.metric("Regime", score_data.get('regime_change', 'STABLE'))
                
                # Predictive forecast section
                if score_data.get('forecast'):
                    st.markdown("##### 🔮 Predictive Analytics")
                    forecast = score_data['forecast']
                    
                    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
                    with col_f1:
                        st.metric("Next Month Flow", 
                                 f"Rp {forecast.get('forecast', 0)/1e9:.1f}B",
                                 delta=forecast.get('direction', ''))
                    with col_f2:
                        st.metric("Confidence", f"{forecast.get('confidence', 0)*100:.0f}%")
                    with col_f3:
                        st.metric("Trend", forecast.get('trend', '').upper())
                    with col_f4:
                        st.metric("Strength", forecast.get('strength', 'N/A'))
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # TAB 3: MARKET INTELLIGENCE - ENHANCED
    with tab3:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📊 Market Intelligence Dashboard</div>', unsafe_allow_html=True)
        
        latest_dates = sorted(df_merged['Date'].unique(), reverse=True)[:30]
        selected_date = st.selectbox(
            "Select Market Date",
            options=latest_dates,
            format_func=lambda x: x.strftime('%d %b %Y'),
            index=0,
            key="market_date_select"
        )
        
        if selected_date:
            daily_data = df_merged[df_merged['Date'] == selected_date]
            
            if not daily_data.empty:
                # Market stats with enhanced visualization
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    advancers = (daily_data['Change %'] > 0).sum()
                    decliners = (daily_data['Change %'] < 0).sum()
                    ratio = advancers / max(decliners, 1)
                    ratio_color = "#05CD99" if ratio > 1 else "#EE5D50"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size:0.9rem; color:#6c757d; margin-bottom:0.5rem;">Market Breadth</div>
                        <div style="font-size:1.5rem; font-weight:700;">{advancers}/{decliners}</div>
                        <div style="font-size:0.8rem; color:{ratio_color}; margin-top:0.5rem;">
                            Adv/Dec Ratio: {ratio:.2f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_m2:
                    total_value = daily_data['Value'].sum() / 1e12
                    avg_value = daily_data['Value'].mean() / 1e9
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size:0.9rem; color:#6c757d; margin-bottom:0.5rem;">Trading Value</div>
                        <div style="font-size:1.5rem; font-weight:700;">Rp {total_value:.2f}T</div>
                        <div style="font-size:0.8rem; color:#A3AED0; margin-top:0.5rem;">
                            Avg: Rp {avg_value:.1f}B
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_m3:
                    avg_change = daily_data['Change %'].mean()
                    median_change = daily_data['Change %'].median()
                    change_color = "#05CD99" if avg_change > 0 else "#EE5D50"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size:0.9rem; color:#6c757d; margin-bottom:0.5rem;">Price Changes</div>
                        <div style="font-size:1.5rem; font-weight:700; color:{change_color};">{avg_change:.2f}%</div>
                        <div style="font-size:0.8rem; color:#A3AED0; margin-top:0.5rem;">
                            Median: {median_change:.2f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_m4:
                    unusual = daily_data['Unusual Volume'].sum() if 'Unusual Volume' in daily_data.columns else 0
                    unusual_pct = (unusual / len(daily_data)) * 100 if len(daily_data) > 0 else 0
                    unusual_color = "#FFB547" if unusual_pct > 10 else "#A3AED0"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size:0.9rem; color:#6c757d; margin-bottom:0.5rem;">Unusual Volume</div>
                        <div style="font-size:1.5rem; font-weight:700; color:{unusual_color};">{unusual}</div>
                        <div style="font-size:0.8rem; color:#A3AED0; margin-top:0.5rem;">
                            {unusual_pct:.1f}% of stocks
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Top tables with enhanced visualization
                col_tab1, col_tab2 = st.columns(2)
                
                with col_tab1:
                    st.markdown("##### 📈 Top Gainers")
                    top_gainers = daily_data.nlargest(10, 'Change %')[
                        ['Stock Code', 'Close', 'Change %', 'Value', 'Volume']
                    ].copy()
                    top_gainers['Value_B'] = top_gainers['Value'] / 1e9
                    top_gainers['Volume_M'] = top_gainers['Volume'] / 1e6
                    
                    fig_gain = px.bar(
                        top_gainers,
                        x='Stock Code',
                        y='Change %',
                        color='Change %',
                        color_continuous_scale='Greens',
                        title="",
                        hover_data=['Close', 'Value_B', 'Volume_M']
                    )
                    fig_gain.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_gain, use_container_width=True)
                
                with col_tab2:
                    st.markdown("##### 💰 Top Value Transactions")
                    top_value = daily_data.nlargest(10, 'Value')[
                        ['Stock Code', 'Close', 'Value', 'Change %', 'NFF_Rp']
                    ].copy()
                    top_value['Value_T'] = top_value['Value'] / 1e12
                    top_value['NFF_B'] = top_value['NFF_Rp'] / 1e9
                    
                    fig_value = px.scatter(
                        top_value,
                        x='Stock Code',
                        y='Value_T',
                        size='Value_T',
                        color='Change %',
                        color_continuous_scale='RdYlGn',
                        title="",
                        hover_data=['Close', 'NFF_B']
                    )
                    fig_value.update_layout(height=400)
                    st.plotly_chart(fig_value, use_container_width=True)
                
                # Foreign activity heatmap
                st.markdown("##### 🌍 Foreign Activity by Sector")
                if 'Sector' in daily_data.columns and 'NFF_Rp' in daily_data.columns:
                    sector_foreign = daily_data.groupby('Sector')['NFF_Rp'].sum().reset_index()
                    sector_foreign['NFF_B'] = sector_foreign['NFF_Rp'] / 1e9
                    
                    fig_foreign = px.bar(
                        sector_foreign.sort_values('NFF_B', ascending=False),
                        x='Sector',
                        y='NFF_B',
                        color='NFF_B',
                        color_continuous_scale='RdYlGn',
                        title="Net Foreign Flow by Sector (Billion Rp)"
                    )
                    fig_foreign.update_layout(height=400)
                    st.plotly_chart(fig_foreign, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # TAB 4: SECTOR ROTATION - ENHANCED
    with tab4:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🔄 Sector Rotation Analysis</div>', unsafe_allow_html=True)
        
        col_dr1, col_dr2 = st.columns(2)
        with col_dr1:
            start_date = st.date_input(
                "Start Date",
                value=df_merged['Date'].max().date() - timedelta(days=30),
                key="sector_start"
            )
        with col_dr2:
            end_date = st.date_input(
                "End Date",
                value=df_merged['Date'].max().date(),
                key="sector_end"
            )
        
        if start_date and end_date and start_date <= end_date:
            period_data = df_merged[
                (df_merged['Date'].dt.date >= start_date) & 
                (df_merged['Date'].dt.date <= end_date)
            ]
            
            if not period_data.empty and 'Smart_Money_Flow' in period_data.columns:
                # Use month-end data for sector analysis
                month_ends = period_data[period_data['Date'].dt.is_month_end]
                if not month_ends.empty:
                    sector_flow = month_ends.groupby('Sector').agg({
                        'Smart_Money_Flow': 'sum',
                        'Retail_Flow': 'sum',
                        'Stock Code': 'nunique',
                        'Close': 'last'
                    }).reset_index()
                    
                    sector_flow.columns = ['Sector', 'Smart Money Flow', 'Retail Flow', 'Stock Count', 'Last Price']
                    sector_flow['Net Institutional'] = sector_flow['Smart Money Flow'] - sector_flow['Retail Flow']
                    sector_flow['Smart Money Flow_B'] = sector_flow['Smart Money Flow'] / 1e9
                    sector_flow['Net Institutional_B'] = sector_flow['Net Institutional'] / 1e9
                    
                    # Charts
                    col_sr1, col_sr2 = st.columns(2)
                    
                    with col_sr1:
                        st.markdown("##### 🏆 Top Sector Inflows")
                        top_inflows = sector_flow.nlargest(10, 'Net Institutional_B')
                        fig_inflows = px.bar(
                            top_inflows,
                            x='Sector',
                            y='Net Institutional_B',
                            color='Net Institutional_B',
                            color_continuous_scale='Greens',
                            title="Top Sector Inflows (B)",
                            hover_data=['Stock Count', 'Last Price']
                        )
                        st.plotly_chart(fig_inflows, use_container_width=True)
                    
                    with col_sr2:
                        st.markdown("##### 🔻 Top Sector Outflows")
                        top_outflows = sector_flow.nsmallest(10, 'Net Institutional_B')
                        fig_outflows = px.bar(
                            top_outflows,
                            x='Sector',
                            y='Net Institutional_B',
                            color='Net Institutional_B',
                            color_continuous_scale='Reds',
                            title="Top Sector Outflows (B)",
                            hover_data=['Stock Count', 'Last Price']
                        )
                        st.plotly_chart(fig_outflows, use_container_width=True)
                    
                    # Sector performance matrix
                    st.markdown("##### 📊 Sector Performance Matrix")
                    
                    # Calculate additional metrics
                    sector_perf = period_data.groupby('Sector').agg({
                        'Change %': 'mean',
                        'Value': 'sum',
                        'Volume': 'sum',
                        'Stock Code': 'nunique'
                    }).reset_index()
                    
                    sector_perf.columns = ['Sector', 'Avg Change %', 'Total Value', 'Total Volume', 'Stock Count']
                    sector_perf['Total Value_T'] = sector_perf['Total Value'] / 1e12
                    
                    # Merge with flow data
                    sector_matrix = pd.merge(
                        sector_flow[['Sector', 'Net Institutional_B', 'Stock Count']],
                        sector_perf[['Sector', 'Avg Change %', 'Total Value_T']],
                        on='Sector',
                        how='left'
                    )
                    
                    # Create scatter plot
                    fig_matrix = px.scatter(
                        sector_matrix,
                        x='Net Institutional_B',
                        y='Avg Change %',
                        size='Total Value_T',
                        color='Sector',
                        hover_data=['Stock Count'],
                        title="Sector Performance vs Institutional Flow",
                        labels={
                            'Net Institutional_B': 'Institutional Flow (B)',
                            'Avg Change %': 'Average Price Change (%)',
                            'Total Value_T': 'Trading Value (T)'
                        }
                    )
                    fig_matrix.update_layout(height=500)
                    st.plotly_chart(fig_matrix, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # TAB 5: PREDICTIVE ANALYTICS
    with tab5:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🔮 Predictive Analytics Engine</div>', unsafe_allow_html=True)
        
        st.info("""
        **Predictive Features:**
        - Next month smart money flow forecasting
        - Regime change detection
        - Correlation analysis between factors
        - Risk assessment and probability scoring
        """)
        
        # Stock selection for prediction
        predictive_stocks = sorted(df_merged['Stock Code'].unique())
        selected_predictive = st.selectbox(
            "Select Stock for Prediction",
            predictive_stocks,
            index=predictive_stocks.index('BBCA') if 'BBCA' in predictive_stocks else 0
        )
        
        if selected_predictive:
            predictive = PredictiveAnalytics(df_merged)
            
            col_pred1, col_pred2 = st.columns(2)
            
            with col_pred1:
                # Flow forecasting
                st.markdown("##### 📈 Flow Forecasting")
                forecast = predictive.forecast_next_month_flow(selected_predictive)
                
                if forecast:
                    col_fc1, col_fc2 = st.columns(2)
                    with col_fc1:
                        st.metric("Next Month Flow", 
                                 f"Rp {forecast.get('forecast', 0)/1e9:.1f}B",
                                 delta=forecast.get('direction', ''))
                    with col_fc2:
                        st.metric("Confidence", f"{forecast.get('confidence', 0)*100:.0f}%")
                    
                    st.markdown("**Forecast Details:**")
                    st.write(f"- **Trend:** {forecast.get('trend', '').upper()}")
                    st.write(f"- **Strength:** {forecast.get('strength', 'N/A')}")
                    st.write(f"- **Magnitude Change:** {forecast.get('magnitude_change', 0):.1%}")
                else:
                    st.warning("Insufficient data for forecasting")
            
            with col_pred2:
                # Regime detection
                st.markdown("##### 🔄 Regime Detection")
                regime = predictive.detect_regime_change(selected_predictive)
                
                if regime:
                    regime_icons = {
                        "ACCELERATING_ACCUMULATION": "🚀",
                        "ACCELERATING_DISTRIBUTION": "📉",
                        "INTENSIFYING_ACCUMULATION": "📈",
                        "INTENSIFYING_DISTRIBUTION": "📊",
                        "DECELERATING": "⚡",
                        "STABLE": "⚖️"
                    }
                    
                    icon = regime_icons.get(regime, "📊")
                    st.metric("Current Regime", f"{icon} {regime}")
                    
                    # Interpretation
                    if "ACCELERATING" in regime:
                        st.success("**Alert:** Significant change in accumulation pattern detected!")
                    elif "INTENSIFYING" in regime:
                        st.info("**Note:** Increasing intensity in current trend")
                    elif regime == "STABLE":
                        st.info("**Status:** Stable accumulation pattern")
                else:
                    st.warning("Insufficient data for regime detection")
            
            # Historical pattern analysis
            st.markdown("##### 📊 Historical Pattern Analysis")
            stock_data = df_merged[df_merged['Stock Code'] == selected_predictive].sort_values('Date')
            
            if not stock_data.empty and 'Smart_Money_Flow' in stock_data.columns:
                monthly = stock_data[stock_data['Date'].dt.is_month_end]
                
                if not monthly.empty:
                    fig_pattern = px.line(
                        monthly,
                        x='Date',
                        y='Smart_Money_Flow',
                        title=f"{selected_predictive} - Monthly Smart Money Flow",
                        markers=True
                    )
                    
                    # Add zero line
                    fig_pattern.add_hline(y=0, line_dash="dash", line_color="gray")
                    
                    # Color positive/negative areas
                    fig_pattern.update_traces(
                        line=dict(color='#4318FF', width=3),
                        marker=dict(size=8)
                    )
                    
                    fig_pattern.update_layout(height=400)
                    st.plotly_chart(fig_pattern, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # TAB 6: PORTFOLIO LAB
    with tab6:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">💼 Portfolio Simulation Lab</div>', unsafe_allow_html=True)
        
        # Check if we have gems data
        if 'find_gems' in st.session_state and st.session_state.find_gems:
            if 'filtered_df' in locals() and not filtered_df.empty:
                # Portfolio configuration
                col_port1, col_port2, col_port3 = st.columns(3)
                
                with col_port1:
                    portfolio_size = st.slider("Portfolio Size", 5, 20, 10, 1)
                
                with col_port2:
                    risk_tolerance = st.select_slider(
                        "Risk Tolerance",
                        options=['Conservative', 'Moderate', 'Aggressive'],
                        value='Moderate'
                    )
                
                with col_port3:
                    rebalance_freq = st.selectbox(
                        "Rebalance Frequency",
                        ['Monthly', 'Quarterly', 'Semi-Annually', 'Annually']
                    )
                
                # Simulate portfolio
                if st.button("🚀 Simulate Portfolio", type="primary", use_container_width=True):
                    simulation = simulator.simulate_portfolio(
                        filtered_df, 
                        sim_capital, 
                        portfolio_size
                    )
                    
                    if simulation:
                        st.session_state.portfolio_simulation = simulation
                
                # Display portfolio results
                if 'portfolio_simulation' in st.session_state:
                    sim = st.session_state.portfolio_simulation
                    
                    st.markdown("#### 📊 Portfolio Simulation Results")
                    
                    # Portfolio metrics
                    col_met1, col_met2, col_met3, col_met4 = st.columns(4)
                    with col_met1:
                        st.metric("Total Capital", f"Rp {sim['metrics']['total_capital']/1e9:.1f}B")
                    with col_met2:
                        st.metric("Number of Stocks", sim['metrics']['number_of_stocks'])
                    with col_met3:
                        st.metric("Avg Gem Score", f"{sim['metrics']['average_gem_score']:.1f}")
                    with col_met4:
                        st.metric("Expected Return", f"{sim['metrics']['expected_return']:.1f}%")
                    
                    # Portfolio allocation chart
                    st.markdown("##### 📈 Portfolio Allocation")
                    
                    fig_allocation = px.pie(
                        sim['portfolio'],
                        values='Allocation',
                        names='Stock',
                        title="Portfolio Allocation by Stock",
                        hole=0.4
                    )
                    fig_allocation.update_layout(height=400)
                    st.plotly_chart(fig_allocation, use_container_width=True)
                    
                    # Sector allocation
                    st.markdown("##### 🏭 Sector Allocation")
                    
                    fig_sector = px.bar(
                        sim['sector_allocation'],
                        x='Sector',
                        y='Percentage',
                        color='Percentage',
                        color_continuous_scale='Blues',
                        title="Portfolio Allocation by Sector (%)"
                    )
                    fig_sector.update_layout(height=400)
                    st.plotly_chart(fig_sector, use_container_width=True)
                    
                    # Detailed portfolio table
                    st.markdown("##### 📋 Portfolio Details")
                    
                    display_portfolio = sim['portfolio'].copy()
                    display_portfolio['Allocation_B'] = display_portfolio['Allocation'] / 1e9
                    display_portfolio['Allocation %'] = (display_portfolio['Allocation'] / sim_capital) * 100
                    
                    st.dataframe(
                        display_portfolio[
                            ['Stock', 'Sector', 'Allocation_B', 'Allocation %', 
                             'Gem Score', 'Signal', 'Risk Level', 'Expected Return']
                        ].style.format({
                            'Allocation_B': 'Rp {:.2f}B',
                            'Allocation %': '{:.1f}%',
                            'Gem Score': '{:.1f}',
                            'Expected Return': '{:.1f}%'
                        }),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Export portfolio
                    col_exp1, col_exp2 = st.columns(2)
                    with col_exp1:
                        portfolio_csv = sim['portfolio'].to_csv(index=False)
                        st.download_button(
                            label="📥 Download Portfolio CSV",
                            data=portfolio_csv,
                            file_name=f"portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col_exp2:
                        if st.button("🔄 Rebalance Portfolio", use_container_width=True):
                            st.session_state.rebalance_portfolio = True
            else:
                st.info("Please run the gem analysis first to simulate a portfolio.")
        else:
            st.info("👈 Run the gem analysis in Tab 1 first to enable portfolio simulation.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # TAB 7: DATA DIAGNOSTICS
    with tab7:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📁 Data Diagnostics & Quality</div>', unsafe_allow_html=True)
        
        # Data summary
        col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
        with col_sum1:
            st.metric("Total Rows", f"{data_checks['total_rows']:,}")
        with col_sum2:
            st.metric("Total Columns", df_merged.shape[1])
        with col_sum3:
            st.metric("Missing Data", data_checks['null_percentage'])
        with col_sum4:
            memory_mb = df_merged.memory_usage(deep=True).sum() / 1024 / 1024
            st.metric("Memory Usage", f"{memory_mb:.1f} MB")
        
        # Data quality visualization
        st.markdown("##### 📊 Data Quality Metrics")
        
        # Create quality metrics
        quality_metrics = {
            'KSEI Coverage': float(data_checks['ksei_coverage'].replace('%', '')),
            'Daily Coverage': float(data_checks['daily_coverage'].replace('%', '')),
            'Data Completeness': 100 - float(data_checks['null_percentage'].replace('%', '')),
            'Data Freshness': 100 - min(100, (datetime.now().date() - df_merged['Date'].max().date()).days * 2)
        }
        
        fig_quality = px.bar(
            x=list(quality_metrics.keys()),
            y=list(quality_metrics.values()),
            color=list(quality_metrics.values()),
            color_continuous_scale='RdYlGn',
            title="Data Quality Metrics (%)",
            labels={'x': 'Metric', 'y': 'Score %'}
        )
        fig_quality.update_layout(height=400)
        st.plotly_chart(fig_quality, use_container_width=True)
        
        # Column explorer
        st.markdown("##### 🗂️ Column Information")
        col_info = pd.DataFrame({
            'Column': df_merged.columns,
            'Type': df_merged.dtypes.astype(str),
            'Non-Null': df_merged.count().values,
            'Unique': df_merged.nunique().values,
            'Missing %': (df_merged.isnull().sum().values / len(df_merged) * 100).round(1)
        })
        st.dataframe(
            col_info.style.background_gradient(subset=['Missing %'], cmap='Reds', vmin=0, vmax=100),
            use_container_width=True,
            height=400
        )
        
        # Sample data
        with st.expander("🔍 View Sample Data (First 50 rows)"):
            st.dataframe(df_merged.head(50), use_container_width=True)
        
        # Data quality report
        with st.expander("📊 Comprehensive Data Quality Report"):
            st.write("**Data Coverage:**")
            st.write(f"- KSEI Monthly Data: {data_checks['ksei_coverage']}")
            st.write(f"- Daily Trading Data: {data_checks['daily_coverage']}")
            st.write(f"- Overall Coverage: {100 - float(data_checks['null_percentage'].replace('%', '')):.1f}%")
            
            st.write("**Date Range:**")
            st.write(f"- From: {data_checks['date_range'].split(' to ')[0]}")
            st.write(f"- To: {data_checks['date_range'].split(' to ')[1]}")
            st.write(f"- Total Days: {(df_merged['Date'].max() - df_merged['Date'].min()).days}")
            
            st.write("**Data Integrity:**")
            st.write(f"- Duplicate Records: {data_checks['duplicates']}")
            st.write(f"- Unique Stocks: {data_checks['unique_stocks']}")
            
            # Top stocks by data completeness
            st.write("**Top Stocks by Data Completeness:**")
            completeness = df_merged.groupby('Stock Code').apply(
                lambda x: pd.Series({
                    'KSEI_Data_Points': x['Smart_Money_Flow'].notna().sum(),
                    'Trading_Days': x['Close'].notna().sum(),
                    'Completeness_Score': (x['Smart_Money_Flow'].notna().sum() / max(x['Close'].notna().sum(), 1)) * 100
                })
            ).reset_index()
            
            st.dataframe(
                completeness.nlargest(10, 'Completeness_Score')
                .style.format({
                    'KSEI_Data_Points': '{:.0f}',
                    'Trading_Days': '{:.0f}',
                    'Completeness_Score': '{:.1f}%'
                }),
                use_container_width=True
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # FOOTER
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #A3AED0; font-size: 14px; padding: 2rem;'>"
        "💎 HIDDEN GEM FINDER v3.0 • Enterprise Edition • "
        f"Data as of {df_merged['Date'].max().strftime('%d %b %Y')} • "
        f"Analysis Period: {lookback_days} days • "
        "© 2024 Hidden Gem Analytics"
        "</div>",
        unsafe_allow_html=True
    )


# ==============================================================================
# 🚀 RUN APPLICATION
# ==============================================================================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.info("Please try refreshing the page or contact support if the issue persists.")
        st.code(f"Error details: {e}", language="python")
