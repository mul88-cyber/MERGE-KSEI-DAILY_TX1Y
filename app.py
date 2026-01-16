# ==============================================================================
# 🚀 FINAL: HIDDEN GEM FINDER - KSEI MONTHLY + DAILY TX1Y MERGE
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
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Google Drive imports
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ==============================================================================
# ⚙️ PAGE CONFIG & CSS
# ==============================================================================
st.set_page_config(
    page_title="💎 HIDDEN GEM FINDER - KSEI & TX1Y",
    layout="wide",
    page_icon="💎",
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
    .gem-badge {
        background: linear-gradient(90deg, #FFD700 0%, #FFA500 100%);
        color: #2B3674;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 12px;
        display: inline-block;
        margin: 2px;
    }
    .signal-buy { background-color: #D1FAE5; color: #065F46; padding: 4px 8px; border-radius: 10px; font-weight: bold; }
    .signal-sell { background-color: #FEE2E2; color: #991B1B; padding: 4px 8px; border-radius: 10px; font-weight: bold; }
    .signal-neutral { background-color: #E5E7EB; color: #4B5563; padding: 4px 8px; border-radius: 10px; font-weight: bold; }
    .data-point-monthly { background-color: #DBEAFE; color: #1E40AF; padding: 2px 6px; border-radius: 6px; font-size: 11px; }
    .data-point-daily { background-color: #D1FAE5; color: #065F46; padding: 2px 6px; border-radius: 6px; font-size: 11px; }
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

# ==============================================================================
# 📦 DATA LOADER CLASS - MONTHLY + DAILY MERGE
# ==============================================================================
class DataLoader:
    def __init__(self):
        self.service = None
        self.initialize_gdrive()
    
    def parse_creds_from_secrets(self, creds_data):
        """Robust parsing of credentials from Streamlit secrets"""
        try:
            if isinstance(creds_data, dict):
                return creds_data
            
            if isinstance(creds_data, str):
                creds_str = creds_data.strip()
                
                if (creds_str.startswith("'") and creds_str.endswith("'")) or \
                   (creds_str.startswith('"') and creds_str.endswith('"')):
                    creds_str = creds_str[1:-1]
                
                if creds_str.startswith("'''") and creds_str.endswith("'''"):
                    creds_str = creds_str[3:-3]
                elif creds_str.startswith('"""') and creds_str.endswith('"""'):
                    creds_str = creds_str[3:-3]
                
                creds_str = creds_str.replace('\\n', '\n').replace('\\\\n', '\n')
                creds_str = creds_str.replace('\\"', '"').replace("\\'", "'")
                
                try:
                    return json.loads(creds_str)
                except json.JSONDecodeError:
                    try:
                        import ast
                        return ast.literal_eval(creds_str)
                    except:
                        import re
                        json_match = re.search(r'\{.*\}', creds_str, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(0)
                            json_str = json_str.replace('\n', '\\n')
                            return json.loads(json_str)
                        else:
                            raise ValueError("Could not parse credentials string")
            
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
            if "gcp_service_account" not in st.secrets:
                st.error("❌ 'gcp_service_account' not found in secrets.toml")
                return False
            
            creds_data = st.secrets["gcp_service_account"]
            creds_json = self.parse_creds_from_secrets(creds_data)
            if creds_json is None:
                return False
            
            creds = Credentials.from_service_account_info(
                creds_json, 
                scopes=['https://www.googleapis.com/auth/drive.readonly']
            )
            
            self.service = build('drive', 'v3', credentials=creds, cache_discovery=False)
            
            try:
                about = self.service.about().get(fields="user").execute()
                st.success(f"✅ Google Drive authenticated: {about.get('user', {}).get('emailAddress', 'Service Account')}")
            except Exception as test_err:
                st.warning(f"⚠️ Auth test: {test_err}")
            
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
            query = f"'{FOLDER_ID}' in parents and name='{file_name}' and trashed=false"
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
                status, done = downloader.next_chunk()
            
            fh.seek(0)
            return fh, None
            
        except Exception as e:
            return None, f"Download error: {e}"
    
    @st.cache_data(ttl=3600, show_spinner="📅 Loading Monthly KSEI Data...")
    def load_ksei_data(_self):
        """Load and process MONTHLY KSEI ownership data"""
        fh, error = _self.download_file(FILE_KSEI)
        if error:
            st.error(f"KSEI: {error}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(fh, dtype=object)
            st.write(f"📅 KSEI Monthly Data: {df.shape[0]} rows loaded")
            
            df.columns = df.columns.str.strip()
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df[df['Date'].dt.year >= 2023].copy()
            
            numeric_cols = OWNERSHIP_COLS + OWNERSHIP_CHG_COLS + OWNERSHIP_CHG_RP_COLS + ['Price', 'Free Float', 'Total_Local', 'Total_Foreign']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            
            local_cols = [c for c in OWNERSHIP_CHG_RP_COLS if 'Local' in c and c in df.columns]
            foreign_cols = [c for c in OWNERSHIP_CHG_RP_COLS if 'Foreign' in c and c in df.columns]
            
            df['Total_Local_chg_Rp'] = df[local_cols].sum(axis=1) if local_cols else 0
            df['Total_Foreign_chg_Rp'] = df[foreign_cols].sum(axis=1) if foreign_cols else 0
            df['Total_chg_Rp'] = df['Total_Local_chg_Rp'] + df['Total_Foreign_chg_Rp']
            
            smart_cols = [c for c in SMART_MONEY_COLS if c in df.columns]
            df['Smart_Money_Flow'] = df[smart_cols].sum(axis=1) if smart_cols else 0
            
            retail_cols = [c for c in RETAIL_COLS if c in df.columns]
            df['Retail_Flow'] = df[retail_cols].sum(axis=1) if retail_cols else 0
            
            df['Institutional_Net'] = df['Smart_Money_Flow'] - df['Retail_Flow']
            df['Stock Code'] = df['Code']
            
            if 'Free Float' in df.columns:
                df['Ownership_Concentration'] = 100 - df['Free Float']
            
            st.success(f"✅ KSEI Monthly: {df['Date'].min().date()} to {df['Date'].max().date()}")
            return df
            
        except Exception as e:
            st.error(f"Error processing KSEI data: {e}")
            return pd.DataFrame()
    
    @st.cache_data(ttl=3600, show_spinner="📈 Loading Daily Historical Data...")
    def load_historical_data(_self):
        """Load and process DAILY historical data"""
        fh, error = _self.download_file(FILE_HIST)
        if error:
            st.error(f"Historical: {error}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(fh, dtype=object)
            st.write(f"📈 Daily Historical Data: {df.shape[0]} rows loaded")
            
            df.columns = df.columns.str.strip()
            df['Last Trading Date'] = pd.to_datetime(df['Last Trading Date'], errors='coerce')
            df['Date'] = df['Last Trading Date']
            
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
            
            if 'Typical Price' in df.columns:
                df['NFF_Rp'] = df['Net Foreign Flow'] * df['Typical Price']
            else:
                df['NFF_Rp'] = df['Net Foreign Flow'] * df['Close']
            
            if 'Unusual Volume' in df.columns:
                df['Unusual Volume'] = df['Unusual Volume'].astype(str).str.strip().str.lower().isin(['spike volume signifikan', 'true'])
            
            if 'Sector' in df.columns:
                df['Sector'] = df['Sector'].astype(str).str.strip().fillna('Others')
            else:
                df['Sector'] = 'Others'
            
            if 'Close' in df.columns:
                df['Price_MA20'] = df.groupby('Stock Code')['Close'].transform(lambda x: x.rolling(20, min_periods=5).mean())
                df['Price_MA50'] = df.groupby('Stock Code')['Close'].transform(lambda x: x.rolling(50, min_periods=10).mean())
                
                def calculate_rsi(prices, period=14):
                    if len(prices) < period + 1:
                        return pd.Series([50] * len(prices), index=prices.index)
                    delta = prices.diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
                    loss = loss.replace(0, np.nan)
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    rsi = rsi.fillna(50).clip(0, 100)
                    return rsi
                
                df['RSI_14'] = df.groupby('Stock Code')['Close'].transform(calculate_rsi)
            
            st.success(f"✅ Daily Historical: {df['Date'].min().date()} to {df['Date'].max().date()}")
            return df
            
        except Exception as e:
            st.error(f"Error processing historical data: {e}")
            return pd.DataFrame()
    
    @st.cache_data(ttl=3600, show_spinner="🔄 Merging Monthly KSEI + Daily Data...")
    def load_merged_data(_self):
        """Intelligent merge of MONTHLY KSEI + DAILY historical data (FIXED)"""
        df_ksei = _self.load_ksei_data()
        df_hist = _self.load_historical_data()
        
        if df_ksei.empty or df_hist.empty:
            return pd.DataFrame()
        
        try:
            # Prepare data
            df_ksei_m = df_ksei.copy()
            df_hist_m = df_hist.copy()
            
            df_ksei_m['Date'] = pd.to_datetime(df_ksei_m['Date'])
            df_hist_m['Date'] = pd.to_datetime(df_hist_m['Date'])
            
            # Create complete date-stock grid to handle gaps
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
            )
            
            merged = merged.sort_values(['Stock Code', 'Date'])
            
            # KSEI columns to forward fill (MONTHLY -> DAILY)
            ksei_cols = [
                'Total_chg_Rp', 'Smart_Money_Flow', 'Retail_Flow', 
                'Institutional_Net', 'Free Float', 'Sector',
                'Ownership_Concentration', 'Price'
            ]
            
            # Filter columns that actually exist in KSEI data
            ksei_cols = [col for col in ksei_cols if col in df_ksei_m.columns]
            
            # Forward fill monthly KSEI data to daily
            for col in ksei_cols:
                # Ambil subset data KSEI
                temp_ksei = df_ksei_m[['Date', 'Stock Code', col]].copy()
                temp_ksei = temp_ksei.dropna(subset=[col])
                
                # --- FIX: Rename Explicitly Before Merge ---
                # Ini mencegah error jika kolom tidak ada di tabel kiri
                raw_col_name = f'{col}_ksei_raw'
                temp_ksei = temp_ksei.rename(columns={col: raw_col_name})
                
                merged = pd.merge(
                    merged,
                    temp_ksei,
                    on=['Date', 'Stock Code'],
                    how='left'
                )
                
                # Forward fill logic
                merged[col] = merged.groupby('Stock Code')[raw_col_name].ffill()
                
                # Cleanup raw column
                if raw_col_name in merged.columns:
                    merged = merged.drop(columns=[raw_col_name])
            
            # Fill missing Close with KSEI Price if available
            if 'Price' in merged.columns and 'Close' in merged.columns:
                merged['Close'] = merged['Close'].fillna(merged['Price'])
            
            # Calculate derived metrics
            merged['Price_Change_1D'] = merged.groupby('Stock Code')['Close'].pct_change()
            
            if 'Volume' in merged.columns:
                merged['Volume_Change_1D'] = merged.groupby('Stock Code')['Volume'].pct_change()
            
            if 'Money Flow Value' in merged.columns and 'Value' in merged.columns:
                merged['MF_Strength'] = merged['Money Flow Value'] / merged['Value'].replace(0, 1)
            
            # Remove rows with completely no trading data & no ksei data to save memory
            merged = merged.dropna(subset=['Close', 'Smart_Money_Flow'], how='all')
            
            # Add data type flags
            merged['Has_KSEI_Data'] = merged['Smart_Money_Flow'].notna()
            merged['Has_Daily_Data'] = merged['Close'].notna()
            
            st.success(f"✅ MERGE COMPLETE: {merged.shape[0]:,} rows")
            
            return merged
            
        except Exception as e:
            st.error(f"Merge error detail: {str(e)}")
            return pd.DataFrame()

# ==============================================================================
# 🎯 HIDDEN GEM ANALYZER - MONTHLY + DAILY INTELLIGENCE
# ==============================================================================
class HiddenGemAnalyzer:
    def __init__(self, df_merged):
        self.df = df_merged.copy()
        self.latest_date = self.df['Date'].max()
        
    def get_monthly_ksei_data(self, stock_data):
        """Extract and process monthly KSEI data points"""
        # Identify month-end dates (approximate monthly data)
        stock_data = stock_data.copy()
        stock_data['Month'] = stock_data['Date'].dt.to_period('M')
        
        # Get unique monthly records (take last record of each month)
        monthly_data = stock_data.dropna(subset=['Smart_Money_Flow']).copy()
        if not monthly_data.empty:
            monthly_data = monthly_data.sort_values('Date').groupby('Month').last().reset_index()
        
        return monthly_data
    
    def calculate_gem_score(self, stock_code, lookback_days=90):
        """Calculate Hidden Gem Score with MONTHLY KSEI + DAILY technical"""
        try:
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
            
            # 1. SMART MONEY ACCUMULATION (45%) - MONTHLY KSEI DATA
            sm_score = 0
            if not monthly_data.empty and 'Smart_Money_Flow' in monthly_data.columns:
                sm_total = monthly_data['Smart_Money_Flow'].sum()
                positive_months = (monthly_data['Smart_Money_Flow'] > 0).sum()
                total_months = len(monthly_data)
                
                # Amount score (0-40 points)
                amount_score = min(40, (abs(sm_total) / 5e9) * 20) if abs(sm_total) > 0 else 0
                
                # Consistency score (0-30 points)
                consistency_score = (positive_months / max(total_months, 1)) * 30
                
                # Trend score (0-15 points)
                if len(monthly_data) >= 2:
                    sm_values = monthly_data['Smart_Money_Flow'].values
                    if len(sm_values) >= 3:
                        trend = np.polyfit(range(len(sm_values)), sm_values, 1)[0]
                        trend_score = min(15, max(0, trend / 1e9 * 5))
                    else:
                        trend_score = 10 if sm_values[-1] > sm_values[0] else 0
                else:
                    trend_score = 0
                
                # Retail divergence bonus (0-15 points)
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
                
                sm_score = amount_score + consistency_score + trend_score + divergence_score
            
            scores['smart_money'] = min(100, sm_score)
            
            # 2. TECHNICAL ANALYSIS (35%) - DAILY DATA
            tech_score = 50
            
            if 'Close' in recent_data.columns:
                price_change = (latest['Close'] - recent_data.iloc[0]['Close']) / recent_data.iloc[0]['Close'] * 100
                
                # Price position score (-20 to +30)
                if -15 <= price_change <= 25:
                    price_score = 30
                elif price_change < -15:
                    price_score = max(10, 40 + price_change)
                else:
                    price_score = max(0, 40 - price_change)
                
                # Volume trend (0-20)
                if 'Volume' in recent_data.columns:
                    avg_volume = recent_data['Volume'].mean()
                    recent_avg = recent_data['Volume'].tail(10).mean()
                    volume_ratio = recent_avg / avg_volume if avg_volume > 0 else 1
                    volume_score = min(20, (volume_ratio - 1) * 20)
                else:
                    volume_score = 10
                
                # RSI score (0-15)
                if 'RSI_14' in latest:
                    rsi = latest['RSI_14']
                    if 30 <= rsi <= 40:
                        rsi_score = 15
                    elif rsi < 30:
                        rsi_score = 20
                    elif 40 < rsi < 70:
                        rsi_score = 10
                    else:
                        rsi_score = 5
                else:
                    rsi_score = 8
                
                # Moving average alignment (0-10)
                if 'Price_MA20' in latest and 'Price_MA50' in latest:
                    if latest['Close'] > latest['Price_MA20'] > latest['Price_MA50']:
                        ma_score = 10
                    elif latest['Close'] > latest['Price_MA20']:
                        ma_score = 7
                    else:
                        ma_score = 3
                else:
                    ma_score = 5
                
                tech_score = price_score + volume_score + rsi_score + ma_score
            
            scores['technical'] = min(100, max(0, tech_score))
            
            # 3. FUNDAMENTAL & STRUCTURAL (20%)
            funda_score = 50
            
            # Free Float analysis (0-25)
            if 'Free Float' in latest:
                ff = latest['Free Float']
                if 20 <= ff <= 40:
                    ff_score = 25
                elif ff < 20:
                    ff_score = 20
                elif ff < 10:
                    ff_score = 15
                elif ff > 60:
                    ff_score = 5
                else:
                    ff_score = 15
            else:
                ff_score = 10
            
            # Liquidity score (0-15)
            if 'Value' in recent_data.columns:
                avg_daily_value = recent_data['Value'].mean()
                if avg_daily_value > 50e9:
                    liquidity_score = 15
                elif avg_daily_value > 20e9:
                    liquidity_score = 12
                elif avg_daily_value > 5e9:
                    liquidity_score = 10
                else:
                    liquidity_score = 5
            else:
                liquidity_score = 8
            
            # Sector momentum placeholder (0-10)
            sector_score = 8
            
            funda_score = ff_score + liquidity_score + sector_score
            scores['fundamental'] = min(100, max(0, funda_score))
            
            # Calculate weighted total
            weights = {'smart_money': 0.45, 'technical': 0.35, 'fundamental': 0.20}
            total_score = (
                scores['smart_money'] * weights['smart_money'] +
                scores['technical'] * weights['technical'] +
                scores['fundamental'] * weights['fundamental']
            )
            
            # Risk adjustment
            risk_adjust = 1.0
            if 'Institutional_Net' in latest:
                if latest['Institutional_Net'] > 10e9:
                    risk_adjust *= 1.1
                elif latest['Institutional_Net'] < -5e9:
                    risk_adjust *= 0.9
            
            final_score = min(100, max(0, total_score * risk_adjust))
            
            # Determine signal
            if final_score >= 80:
                signal = "💎 STRONG BUY"
                signal_color = "buy"
            elif final_score >= 70:
                signal = "🔥 ACCUMULATE"
                signal_color = "buy"
            elif final_score >= 60:
                signal = "📈 WATCH"
                signal_color = "neutral"
            elif final_score >= 50:
                signal = "⚖️ NEUTRAL"
                signal_color = "neutral"
            else:
                signal = "⏸️ AVOID"
                signal_color = "sell"
            
            return {
                'total_score': round(final_score, 1),
                'component_scores': scores,
                'signal': signal,
                'signal_color': signal_color,
                'latest_price': latest.get('Close', latest.get('Price', 0)),
                'price_change_period': price_change if 'price_change' in locals() else 0,
                'sector': latest.get('Sector', 'N/A'),
                'free_float': latest.get('Free Float', 0),
                'smart_money_total': sm_total if 'sm_total' in locals() else 0,
                'positive_months': positive_months if 'positive_months' in locals() else 0,
                'total_months': total_months if 'total_months' in locals() else 0,
                'monthly_data_points': len(monthly_data) if 'monthly_data' in locals() else 0,
                'volume_ratio': volume_ratio if 'volume_ratio' in locals() else 1,
                'rsi': latest.get('RSI_14', 50)
            }
            
        except Exception as e:
            st.error(f"Error calculating score for {stock_code}: {e}")
            return {'total_score': 0, 'signal': 'ERROR', 'signal_color': 'neutral'}
    
    def find_top_gems(self, top_n=25, min_score=65, sector_filter=None):
        """Find top hidden gem candidates"""
        unique_stocks = self.df['Stock Code'].unique()
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_stocks = min(150, len(unique_stocks))
        
        for i, stock in enumerate(unique_stocks[:total_stocks]):
            status_text.text(f"🔍 {stock}... ({i+1}/{total_stocks})")
            
            score_data = self.calculate_gem_score(stock)
            if score_data['total_score'] >= min_score:
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
                    'Volume Trend': score_data.get('volume_ratio', 1),
                    'Smart Score': score_data['component_scores'].get('smart_money', 0),
                    'Tech Score': score_data['component_scores'].get('technical', 0),
                    'Fundamental Score': score_data['component_scores'].get('fundamental', 0)
                })
            
            progress_bar.progress((i + 1) / total_stocks)
        
        progress_bar.empty()
        status_text.empty()
        
        if results:
            df_results = pd.DataFrame(results)
            df_results = df_results.sort_values('Gem Score', ascending=False).head(top_n)
            return df_results
        else:
            return pd.DataFrame()

# ==============================================================================
# 📊 VISUALIZATION FUNCTIONS
# ==============================================================================
def create_gem_radar_chart(scores, stock_code, signal):
    """Create radar chart for gem score components"""
    categories = ['Smart Money', 'Technical', 'Fundamental']
    values = [
        scores.get('smart_money', 0),
        scores.get('technical', 0),
        scores.get('fundamental', 0)
    ]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(255, 215, 0, 0.3)',
        line=dict(color='#FFD700', width=3)
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100]),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=False,
        title=f"💎 {stock_code} - {signal}",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#2B3674'),
        height=400
    )
    
    return fig

def create_ownership_timeline(df, stock_code):
    """Create timeline of ownership changes (FIXED VERSION)"""
    stock_data = df[df['Stock Code'] == stock_code].sort_values('Date')
    
    if stock_data.empty:
        return None
    
    # 1. Definisikan Layout Subplot dengan Benar
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            f'{stock_code} - Price History', 
            'Smart Money Flow (B) - Monthly', 
            'Institutional vs Retail Net Flow (B)'
        ),
        vertical_spacing=0.08,
        row_heights=[0.5, 0.25, 0.25], # Harga lebih besar
        shared_xaxes=True # Sumbu X sinkron
    )
    
    # --- ROW 1: PRICE CHART ---
    fig.add_trace(
        go.Scatter(
            x=stock_data['Date'], 
            y=stock_data['Close'], 
            name='Price', 
            line=dict(color='#4318FF', width=2),
            mode='lines'
        ),
        row=1, col=1
    )
    
    # --- ROW 2: SMART MONEY (MONTHLY) ---
    if 'Smart_Money_Flow' in stock_data.columns:
        # Ambil data akhir bulan saja agar grafik tidak 'berisik'
        month_ends = stock_data[stock_data['Date'].dt.is_month_end]
        if not month_ends.empty:
            colors = ['#05CD99' if x > 0 else '#EE5D50' for x in month_ends['Smart_Money_Flow']]
            fig.add_trace(
                go.Bar(
                    x=month_ends['Date'], 
                    y=month_ends['Smart_Money_Flow'] / 1e9, 
                    name='Smart Money (B)', 
                    marker_color=colors,
                    opacity=0.8
                ),
                row=2, col=1 # PASTIKAN INI ROW 2, BUKAN ROW 1
            )
    
    # --- ROW 3: INSTITUTIONAL VS RETAIL ---
    # Institutional Bar
    if 'Institutional_Net' in stock_data.columns:
        month_ends_inst = stock_data[stock_data['Date'].dt.is_month_end]
        if not month_ends_inst.empty:
            fig.add_trace(
                go.Bar(
                    x=month_ends_inst['Date'], 
                    y=month_ends_inst['Institutional_Net'] / 1e9, 
                    name='Institusi Net (B)', 
                    marker_color='#4318FF',
                    opacity=0.6
                ),
                row=3, col=1
            )
    
    # Retail Line
    if 'Retail_Flow' in stock_data.columns:
        month_ends_ret = stock_data[stock_data['Date'].dt.is_month_end]
        if not month_ends_ret.empty:
            fig.add_trace(
                go.Scatter(
                    x=month_ends_ret['Date'], 
                    y=month_ends_ret['Retail_Flow'] / 1e9, 
                    name='Retail Flow (B)', 
                    line=dict(color='#EE5D50', width=2, dash='dot'),
                    mode='lines+markers'
                ),
                row=3, col=1
            )
    
    # Update Layout
    fig.update_layout(
        height=700,
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#2B3674'),
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=40, b=10)
    )
    
    # Grid Styling
    fig.update_xaxes(showgrid=True, gridcolor='#E0E5F2')
    fig.update_yaxes(showgrid=True, gridcolor='#E0E5F2')
    
    return fig

def create_sector_heatmap(gems_df):
    """Create heatmap of gems by sector"""
    if gems_df.empty or 'Sector' not in gems_df.columns:
        return None
    
    sector_stats = gems_df.groupby('Sector').agg({
        'Stock': 'count',
        'Gem Score': 'mean',
        'Smart Money (B)': 'sum'
    }).reset_index()
    
    sector_stats.columns = ['Sector', 'Count', 'Avg Score', 'Total Smart Money (B)']
    
    fig = px.treemap(
        sector_stats,
        path=['Sector'],
        values='Total Smart Money (B)',
        color='Avg Score',
        color_continuous_scale='RdYlGn',
        title='💎 Hidden Gems by Sector',
        hover_data=['Count', 'Avg Score']
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#2B3674')
    )
    
    return fig

# ==============================================================================
# 🎨 MAIN DASHBOARD - ALL FEATURES
# ==============================================================================
def main():
    # HEADER
    st.markdown("""
    <div class="header-banner">
        <div class="header-title">🚀 HIDDEN GEM FINDER v2.0</div>
        <div class="header-subtitle">Monthly KSEI Ownership + Daily Technical Analysis • Multi-dimensional Scoring</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize data loader
    loader = DataLoader()
    
    if not loader.service:
        st.error("❌ Failed to initialize Google Drive service")
        return
    
    # Load data
    with st.spinner("🚀 Loading Monthly KSEI + Daily Historical Data..."):
        df_merged = loader.load_merged_data()
    
    if df_merged.empty:
        st.error("Failed to load data. Please check data files.")
        return
    
    # Store in session
    analyzer = HiddenGemAnalyzer(df_merged)
    st.session_state.analyzer = analyzer
    st.session_state.df_merged = df_merged
    
    # SIDEBAR
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=60)
        st.markdown("<h3 style='color:#2B3674;'>💎 Gem Finder v2.0</h3>", unsafe_allow_html=True)
        st.divider()
        
        # Quick stats
        st.markdown("##### 📊 Data Overview")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric("Stocks", df_merged['Stock Code'].nunique())
        with col_s2:
            st.metric("Latest", df_merged['Date'].max().strftime('%d/%m'))
        
        st.divider()
        
        # Gem finder settings
        st.markdown("##### ⚙️ Analysis Settings")
        lookback_days = st.slider("Analysis Period", 30, 180, 90, 15)
        min_gem_score = st.slider("Minimum Score", 50, 90, 65, 5)
        top_n_gems = st.slider("Top N Results", 10, 50, 25, 5)
        
        # Sector filter - FIX: Handle NaN/Float values before sorting
        # 1. Ambil unique values
        # 2. Convert semua ke string (.astype(str)) untuk mencegah error comparison
        # 3. Ganti 'nan' string dengan 'Others' jika ada
        raw_sectors = df_merged['Sector'].dropna().unique()
        clean_sectors = [str(x) for x in raw_sectors if str(x).lower() != 'nan']

        sectors = ['All'] + sorted(clean_sectors)
        selected_sector = st.selectbox("Sector Filter", sectors)
        
        # Advanced filters
        with st.expander("🔍 Advanced Filters"):
            min_smart_money = st.number_input("Min Smart Money (B)", 0.0, 100.0, 1.0, 0.5)
            max_free_float = st.number_input("Max Free Float %", 0.0, 100.0, 60.0, 5.0)
            min_rsi = st.slider("Min RSI", 0, 100, 30, 5)
            max_rsi = st.slider("Max RSI", 0, 100, 70, 5)
        
        st.divider()
        
        # Actions
        if st.button("🔄 Refresh All Data", use_container_width=True, type="primary"):
            st.cache_data.clear()
            st.rerun()
        
        if st.button("📊 Run Full Analysis", use_container_width=True):
            st.session_state.run_analysis = True
        
        st.divider()
        
        # Data info
        with st.expander("📁 Dataset Info"):
            st.write(f"**Total Rows:** {df_merged.shape[0]:,}")
            st.write(f"**Date Range:** {df_merged['Date'].min().date()} to {df_merged['Date'].max().date()}")
            st.write(f"**KSEI Coverage:** {(df_merged['Smart_Money_Flow'].notna().sum() / len(df_merged) * 100):.1f}%")
            st.write(f"**Data Types:** Monthly KSEI + Daily Trading")
    
    # MAIN CONTENT - TABS
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏆 Top Gems", "📈 Stock Analyzer", "📊 Market Overview", 
        "🔄 Sector Rotation", "📉 Technical Scan", "📁 Data Explorer"
    ])
    
    # TAB 1: TOP GEMS
    with tab1:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">💎 Top Hidden Gem Candidates</div>', unsafe_allow_html=True)
        
        col_t1, col_t2 = st.columns([3, 1])
        with col_t1:
            st.markdown(f"**Period:** {lookback_days} days | **Min Score:** {min_gem_score} | **Sector:** {selected_sector}")
        with col_t2:
            if st.button("🔍 Find Gems Now", type="primary", use_container_width=True):
                st.session_state.find_gems = True
        
        if 'find_gems' in st.session_state and st.session_state.find_gems:
            sector_filter = None if selected_sector == 'All' else selected_sector
            
            with st.spinner(f"Analyzing stocks with advanced scoring..."):
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
                
                if not filtered_df.empty:
                    # Summary metrics
                    col_sm1, col_sm2, col_sm3, col_sm4 = st.columns(4)
                    with col_sm1:
                        st.metric("Total Gems", len(filtered_df))
                    with col_sm2:
                        st.metric("Avg Score", f"{filtered_df['Gem Score'].mean():.1f}")
                    with col_sm3:
                        st.metric("Total Smart Money", f"{filtered_df['Smart Money (B)'].sum():.1f}B")
                    with col_sm4:
                        top_sector = filtered_df['Sector'].mode().iloc[0] if not filtered_df.empty else "N/A"
                        st.metric("Top Sector", top_sector)
                    
                    # Sector heatmap
                    st.markdown("#### 🗺️ Sector Distribution")
                    heatmap = create_sector_heatmap(filtered_df)
                    if heatmap:
                        st.plotly_chart(heatmap, use_container_width=True)
                    
                    # Gems table
                    st.markdown("#### 🏆 Filtered Gem Candidates")
                    
                    display_cols = ['Stock', 'Gem Score', 'Signal', 'Sector', 'Price', 
                                  'Price Chg', 'Free Float %', 'Smart Money (B)', 
                                  'Positive Months', 'Monthly Data', 'RSI', 'Volume Trend']
                    
                    display_df = filtered_df[display_cols].copy()
                    
                    def color_signal(val):
                        if 'BUY' in val:
                            return 'color: #065F46; font-weight: bold;'
                        elif 'ACCUMULATE' in val:
                            return 'color: #D97706; font-weight: bold;'
                        elif 'WATCH' in val:
                            return 'color: #2563EB; font-weight: bold;'
                        else:
                            return 'color: #6B7280;'
                    
                    st.dataframe(
                        display_df.style
                        .format({
                            'Price': '{:,.0f}',
                            'Price Chg': '{:.1f}%',
                            'Free Float %': '{:.1f}%',
                            'Smart Money (B)': '{:.2f}',
                            'RSI': '{:.1f}',
                            'Gem Score': '{:.1f}',
                            'Volume Trend': '{:.2f}x'
                        })
                        .applymap(color_signal, subset=['Signal'])
                        .background_gradient(subset=['Gem Score'], cmap='RdYlGn', vmin=65, vmax=100)
                        .bar(subset=['Smart Money (B)'], color='#05CD99')
                        .bar(subset=['Positive Months'], color='#4318FF'),
                        use_container_width=True,
                        hide_index=True,
                        height=600
                    )
                    
                    # Top gem deep dive
                    if len(filtered_df) > 0:
                        st.markdown("#### 🎯 Top Gem Deep Dive")
                        top_gem = filtered_df.iloc[0]['Stock']
                        score_data = analyzer.calculate_gem_score(top_gem, lookback_days)
                        
                        col_ana1, col_ana2 = st.columns(2)
                        
                        with col_ana1:
                            st.plotly_chart(
                                create_gem_radar_chart(score_data['component_scores'], top_gem, score_data['signal']),
                                use_container_width=True
                            )
                        
                        with col_ana2:
                            st.plotly_chart(
                                create_ownership_timeline(df_merged, top_gem),
                                use_container_width=True
                            )
                        
                        # Detailed analysis
                        with st.expander(f"📊 Detailed Analysis: {top_gem}"):
                            col_d1, col_d2, col_d3, col_d4 = st.columns(4)
                            with col_d1:
                                st.metric("Gem Score", f"{score_data['total_score']:.1f}/100")
                            with col_d2:
                                st.metric("Signal", score_data['signal'])
                            with col_d3:
                                months_pos = f"{score_data['positive_months']}/{score_data['total_months']}"
                                st.metric("Positive Months", months_pos)
                            with col_d4:
                                st.metric("Smart Money Total", f"Rp {score_data['smart_money_total']/1e9:.1f}B")
                            
                            st.markdown("**Data Quality:**")
                            col_q1, col_q2, col_q3 = st.columns(3)
                            with col_q1:
                                st.metric("Monthly Points", score_data['monthly_data_points'])
                            with col_q2:
                                st.metric("Free Float", f"{score_data['free_float']:.1f}%")
                            with col_q3:
                                st.metric("RSI", f"{score_data['rsi']:.1f}")
                else:
                    st.warning("No gems found with current filters. Try relaxing criteria.")
            else:
                st.info("No hidden gems found. Try lowering minimum score.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # TAB 2: STOCK ANALYZER
    with tab2:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🔍 Individual Stock Analysis</div>', unsafe_allow_html=True)
        
        available_stocks = sorted(df_merged['Stock Code'].unique())
        selected_stock = st.selectbox(
            "Select Stock", 
            available_stocks,
            index=available_stocks.index('BBRI') if 'BBRI' in available_stocks else 0
        )
        
        if selected_stock:
            score_data = analyzer.calculate_gem_score(selected_stock, lookback_days)
            
            # Header metrics
            col_h1, col_h2, col_h3, col_h4 = st.columns(4)
            with col_h1:
                st.metric("Gem Score", f"{score_data['total_score']:.1f}/100", 
                         delta=score_data['signal'], delta_color="off")
            with col_h2:
                st.metric("Price", f"Rp {score_data['latest_price']:,.0f}",
                         delta=f"{score_data['price_change_period']:.1f}%")
            with col_h3:
                st.metric("Smart Money", f"Rp {score_data['smart_money_total']/1e9:.1f}B",
                         delta=f"{score_data['positive_months']}/{score_data['total_months']} months")
            with col_h4:
                st.metric("Free Float", f"{score_data['free_float']:.1f}%",
                         delta=score_data['sector'])
            
            # Charts
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.plotly_chart(
                    create_gem_radar_chart(score_data['component_scores'], selected_stock, score_data['signal']),
                    use_container_width=True
                )
            
            with col_c2:
                st.plotly_chart(
                    create_ownership_timeline(df_merged, selected_stock),
                    use_container_width=True
                )
            
            # Data quality info
            st.markdown("##### 📅 Data Quality & Frequency")
            col_q1, col_q2, col_q3, col_q4 = st.columns(4)
            with col_q1:
                st.metric("Monthly Points", score_data['monthly_data_points'])
            with col_q2:
                pos_rate = (score_data['positive_months'] / max(score_data['total_months'], 1)) * 100
                st.metric("Positive Rate", f"{pos_rate:.0f}%")
            with col_q3:
                st.metric("RSI", f"{score_data['rsi']:.1f}")
            with col_q4:
                st.metric("Volume Trend", f"{score_data.get('volume_ratio', 1):.2f}x")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # TAB 3: MARKET OVERVIEW
    with tab3:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📊 Market Overview</div>', unsafe_allow_html=True)
        
        latest_dates = sorted(df_merged['Date'].unique(), reverse=True)[:30]
        selected_date = st.selectbox(
            "Select Date",
            options=latest_dates,
            format_func=lambda x: x.strftime('%d %b %Y'),
            index=0
        )
        
        if selected_date:
            daily_data = df_merged[df_merged['Date'] == selected_date]
            
            if not daily_data.empty:
                # Market stats
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    advancers = (daily_data['Change %'] > 0).sum()
                    decliners = (daily_data['Change %'] < 0).sum()
                    st.metric("Advancers/Decliners", f"{advancers}/{decliners}")
                
                with col_m2:
                    total_value = daily_data['Value'].sum() / 1e12
                    st.metric("Total Value", f"Rp {total_value:.2f}T")
                
                with col_m3:
                    avg_change = daily_data['Change %'].mean()
                    st.metric("Avg Change", f"{avg_change:.2f}%")
                
                with col_m4:
                    unusual = daily_data['Unusual Volume'].sum() if 'Unusual Volume' in daily_data.columns else 0
                    st.metric("Unusual Volume", unusual)
                
                # Top tables
                col_tab1, col_tab2 = st.columns(2)
                
                with col_tab1:
                    st.markdown("##### 📈 Top Gainers")
                    top_gainers = daily_data.nlargest(10, 'Change %')[['Stock Code', 'Close', 'Change %', 'Value']]
                    st.dataframe(
                        top_gainers.style.format({
                            'Close': 'Rp {:,.0f}',
                            'Change %': '{:.2f}%',
                            'Value': 'Rp {:,.0f}'
                        }).background_gradient(subset=['Change %'], cmap='Greens'),
                        use_container_width=True,
                        hide_index=True
                    )
                
                with col_tab2:
                    st.markdown("##### 💰 Top Value")
                    top_value = daily_data.nlargest(10, 'Value')[['Stock Code', 'Value', 'Volume', 'Change %']]
                    top_value['Value_B'] = top_value['Value'] / 1e9
                    st.dataframe(
                        top_value[['Stock Code', 'Value_B', 'Volume', 'Change %']]
                        .style.format({
                            'Value_B': 'Rp {:.1f}B',
                            'Volume': '{:,.0f}',
                            'Change %': '{:.2f}%'
                        }),
                        use_container_width=True,
                        hide_index=True
                    )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # TAB 4: SECTOR ROTATION
    with tab4:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🔄 Sector Rotation Analysis</div>', unsafe_allow_html=True)
        
        col_dr1, col_dr2 = st.columns(2)
        with col_dr1:
            start_date = st.date_input(
                "Start Date",
                value=df_merged['Date'].max().date() - timedelta(days=30)
            )
        with col_dr2:
            end_date = st.date_input(
                "End Date",
                value=df_merged['Date'].max().date()
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
                        'Stock Code': 'nunique'
                    }).reset_index()
                    
                    sector_flow.columns = ['Sector', 'Smart Money Flow', 'Retail Flow', 'Stock Count']
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
                            hover_data=['Stock Count']
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
                            hover_data=['Stock Count']
                        )
                        st.plotly_chart(fig_outflows, use_container_width=True)
                    
                    # Sector matrix
                    st.markdown("##### 🔥 Sector Flow Matrix")
                    st.dataframe(
                        sector_flow[['Sector', 'Smart Money Flow_B', 'Retail Flow', 'Net Institutional_B', 'Stock Count']]
                        .sort_values('Net Institutional_B', ascending=False)
                        .style.format({
                            'Smart Money Flow_B': '{:.1f}B',
                            'Retail Flow': '{:.0f}',
                            'Net Institutional_B': '{:.1f}B'
                        }).background_gradient(subset=['Net Institutional_B'], cmap='RdYlGn'),
                        use_container_width=True,
                        hide_index=True
                    )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # TAB 5: TECHNICAL SCAN
    with tab5:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📉 Technical Scan & Screening</div>', unsafe_allow_html=True)
        
        st.info("""
        **Technical Screening Features:**
        - RSI Oversold/Overbought scan
        - Volume spike detection  
        - Price breakout identification
        - Moving average crossovers
        - Support/Resistance levels
        """)
        
        # Technical filters
        col_tech1, col_tech2, col_tech3 = st.columns(3)
        with col_tech1:
            rsi_min = st.slider("Min RSI", 0, 100, 30, 5)
            rsi_max = st.slider("Max RSI", 0, 100, 70, 5)
        with col_tech2:
            price_change_min = st.slider("Min Price Change %", -50, 50, -20, 5)
            price_change_max = st.slider("Max Price Change %", -50, 50, 30, 5)
        with col_tech3:
            volume_spike = st.slider("Min Volume Spike (x)", 1.0, 5.0, 1.5, 0.1)
            ma_crossover = st.checkbox("MA20 > MA50 Crossover")
        
        if st.button("🔍 Run Technical Scan", type="primary"):
            latest_data = df_merged[df_merged['Date'] == df_merged['Date'].max()]
            
            if not latest_data.empty:
                filtered = latest_data.copy()
                
                # Apply filters
                if 'RSI_14' in filtered.columns:
                    filtered = filtered[(filtered['RSI_14'] >= rsi_min) & (filtered['RSI_14'] <= rsi_max)]
                
                if 'Change %' in filtered.columns:
                    filtered = filtered[(filtered['Change %'] >= price_change_min) & 
                                       (filtered['Change %'] <= price_change_max)]
                
                if 'Volume' in filtered.columns and 'MA20_vol' in filtered.columns:
                    filtered = filtered[filtered['Volume'] > filtered['MA20_vol'] * volume_spike]
                
                if ma_crossover and 'Price_MA20' in filtered.columns and 'Price_MA50' in filtered.columns:
                    filtered = filtered[filtered['Price_MA20'] > filtered['Price_MA50']]
                
                st.success(f"Found {len(filtered)} stocks matching criteria")
                st.dataframe(
                    filtered[['Stock Code', 'Close', 'Change %', 'RSI_14', 'Volume', 'Value']]
                    .sort_values('Change %', ascending=False)
                    .head(20)
                    .style.format({
                        'Close': '{:,.0f}',
                        'Change %': '{:.2f}%',
                        'RSI_14': '{:.1f}',
                        'Volume': '{:,.0f}',
                        'Value': '{:,.0f}'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # TAB 6: DATA EXPLORER
    with tab6:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📁 Data Explorer & Diagnostics</div>', unsafe_allow_html=True)
        
        # Data summary
        col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
        with col_sum1:
            st.metric("Total Rows", f"{df_merged.shape[0]:,}")
        with col_sum2:
            st.metric("Total Columns", df_merged.shape[1])
        with col_sum3:
            missing_pct = (df_merged.isnull().sum().sum() / (len(df_merged) * len(df_merged.columns))) * 100
            st.metric("Missing Data", f"{missing_pct:.1f}%")
        with col_sum4:
            st.metric("Memory Usage", f"{df_merged.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
        
        # Column explorer
        st.markdown("##### 🗂️ Column Information")
        col_info = pd.DataFrame({
            'Column': df_merged.columns,
            'Type': df_merged.dtypes.astype(str),
            'Non-Null': df_merged.count().values,
            'Unique': df_merged.nunique().values,
            'Missing %': (df_merged.isnull().sum().values / len(df_merged) * 100).round(1)
        })
        st.dataframe(col_info, use_container_width=True, height=400)
        
        # Sample data
        with st.expander("🔍 View Sample Data (100 rows)"):
            st.dataframe(df_merged.head(100), use_container_width=True)
        
        # Data quality report
        with st.expander("📊 Data Quality Report"):
            ksei_coverage = (df_merged['Smart_Money_Flow'].notna().sum() / len(df_merged) * 100)
            daily_coverage = (df_merged['Close'].notna().sum() / len(df_merged) * 100)
            
            st.write(f"**KSEI Monthly Data Coverage:** {ksei_coverage:.1f}%")
            st.write(f"**Daily Trading Data Coverage:** {daily_coverage:.1f}%")
            st.write(f"**Date Range:** {df_merged['Date'].min().date()} to {df_merged['Date'].max().date()}")
            st.write(f"**Trading Days:** {(df_merged['Date'].max() - df_merged['Date'].min()).days} days")
            
            # Top stocks by data completeness
            completeness = df_merged.groupby('Stock Code').apply(
                lambda x: pd.Series({
                    'KSEI_Data_Points': x['Smart_Money_Flow'].notna().sum(),
                    'Trading_Days': x['Close'].notna().sum(),
                    'Completeness_Score': (x['Smart_Money_Flow'].notna().sum() / max(x['Close'].notna().sum(), 1)) * 100
                })
            ).reset_index()
            
            st.markdown("##### 🏆 Top Stocks by Data Completeness")
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
        "<div style='text-align: center; color: #A3AED0; font-size: 14px;'>"
        "💎 HIDDEN GEM FINDER v2.0 • Monthly KSEI + Daily TX1Y • "
        f"Data as of {df_merged['Date'].max().strftime('%d %b %Y')} • "
        f"© {datetime.now().year} Hidden Gem Analytics"
        "</div>",
        unsafe_allow_html=True
    )

# ==============================================================================
# 🚀 RUN APP
# ==============================================================================
if __name__ == "__main__":
    main()
