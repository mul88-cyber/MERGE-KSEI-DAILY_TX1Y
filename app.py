# ==============================================================================
# 📦 IMPORTS (Tambahan untuk analisis)
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
# ⚙️ PAGE CONFIG & CSS (SAME)
# ==============================================================================
st.set_page_config(
    page_title="🚀 HIDDEN GEM FINDER - KSEI & TX1Y MERGE",
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
# 📦 DATA LOADER CLASS (SAME AS BEFORE - WORKING VERSION)
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
            
            # Calculate ownership concentration
            if 'Free Float' in df.columns:
                df['Ownership_Concentration'] = 100 - df['Free Float']  # Higher = more concentrated
            
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
            
            # Calculate technical indicators
            if 'Close' in df.columns:
                df['Price_MA20'] = df.groupby('Stock Code')['Close'].transform(lambda x: x.rolling(20, min_periods=5).mean())
                df['Price_MA50'] = df.groupby('Stock Code')['Close'].transform(lambda x: x.rolling(50, min_periods=10).mean())
                df['RSI_14'] = df.groupby('Stock Code')['Close'].transform(
                    lambda x: self.calculate_rsi(x, period=14) if len(x) >= 14 else 50
                )
            
            return df
            
        except Exception as e:
            st.error(f"Error processing historical data: {e}")
            return pd.DataFrame()
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 50
        
        deltas = np.diff(prices)
        seed = deltas[:period]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 0
        rsi = 100.0 - 100.0 / (1.0 + rs)
        
        for i in range(period, len(deltas)):
            delta = deltas[i]
            if delta > 0:
                upval = delta
                downval = 0.0
            else:
                upval = 0.0
                downval = -delta
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down if down != 0 else 0
            rsi = 100.0 - 100.0 / (1.0 + rs)
        
        return rsi
    
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
                          'Retail_Flow', 'Institutional_Net', 'Free Float', 'Sector',
                          'Ownership_Concentration'] + 
                         [c for c in OWNERSHIP_COLS if c in df_ksei_m.columns]],
                on=['Date', 'Stock Code'],
                how='left',
                suffixes=('', '_ksei')
            )
            
            # Forward fill KSEI data for continuity
            ksei_cols = ['Total_chg_Rp', 'Smart_Money_Flow', 'Retail_Flow', 'Institutional_Net', 
                        'Free Float', 'Ownership_Concentration']
            merged = merged.sort_values(['Stock Code', 'Date'])
            
            for col in ksei_cols:
                if col in merged.columns:
                    merged[col] = merged.groupby('Stock Code')[col].ffill()
            
            # Calculate derived metrics
            merged['Price_Change_1D'] = merged.groupby('Stock Code')['Close'].pct_change()
            merged['Volume_Change_1D'] = merged.groupby('Stock Code')['Volume'].pct_change()
            
            if 'Money Flow Value' in merged.columns:
                merged['MF_Strength'] = merged['Money Flow Value'] / merged['Value'].replace(0, 1)
            
            # Calculate trend indicators
            merged['Price_Trend_20D'] = merged.groupby('Stock Code')['Close'].transform(
                lambda x: (x.iloc[-1] / x.rolling(20).mean().iloc[-1] - 1) * 100 if len(x) >= 20 else 0
            )
            
            # Calculate Smart Money Accumulation Score
            merged['SM_Accum_Score'] = merged.groupby('Stock Code')['Smart_Money_Flow'].transform(
                lambda x: x.rolling(30, min_periods=10).sum() / 1e9  # In billions
            )
            
            st.success(f"✅ Merged dataset loaded: {merged.shape[0]:,} rows, {merged.shape[1]:,} columns")
            return merged
            
        except Exception as e:
            st.error(f"Merge error: {e}")
            return pd.DataFrame()

# ==============================================================================
# 🎯 HIDDEN GEM ANALYZER
# ==============================================================================
class HiddenGemAnalyzer:
    def __init__(self, df_merged):
        self.df = df_merged.copy()
        self.latest_date = self.df['Date'].max()
    
    def calculate_gem_score(self, stock_code, lookback_days=60):
        """Calculate Hidden Gem Score (0-100) for a stock"""
        try:
            # Get data for this stock
            stock_data = self.df[self.df['Stock Code'] == stock_code].sort_values('Date')
            if stock_data.empty:
                return 0
            
            # Get recent data
            cutoff_date = self.latest_date - timedelta(days=lookback_days)
            recent_data = stock_data[stock_data['Date'] >= cutoff_date]
            
            if recent_data.empty or len(recent_data) < 10:
                return 0
            
            latest = recent_data.iloc[-1]
            
            scores = {}
            
            # 1. SMART MONEY ACCUMULATION (40%)
            sm_score = 0
            if 'Smart_Money_Flow' in recent_data.columns:
                # Total smart money inflow
                sm_total = recent_data['Smart_Money_Flow'].sum()
                
                # Recent accumulation trend (last 10 days vs previous 10 days)
                if len(recent_data) >= 20:
                    sm_recent = recent_data['Smart_Money_Flow'].tail(10).sum()
                    sm_previous = recent_data['Smart_Money_Flow'].iloc[-20:-10].sum()
                    sm_trend = (sm_recent - sm_previous) / max(abs(sm_previous), 1e9) * 100
                else:
                    sm_trend = 0
                
                # Score based on amount and trend
                amount_score = min(100, abs(sm_total) / 5e9 * 20)  # 5B = 20 points
                trend_score = min(30, max(0, sm_trend))  # Up to 30 points for positive trend
                
                # Bonus for consistency (more than 70% days positive)
                positive_days = (recent_data['Smart_Money_Flow'] > 0).sum()
                consistency_score = (positive_days / len(recent_data)) * 20  # Up to 20 points
                
                sm_score = amount_score + trend_score + consistency_score
            
            scores['smart_money'] = min(100, sm_score)
            
            # 2. TECHNICAL DIVERGENCE (30%)
            tech_score = 50  # Base
            
            # Price vs Smart Money divergence
            if 'Close' in recent_data.columns and 'Smart_Money_Flow' in recent_data.columns:
                price_change = (latest['Close'] - recent_data.iloc[0]['Close']) / recent_data.iloc[0]['Close'] * 100
                sm_change = recent_data['Smart_Money_Flow'].sum() / 1e9  # in billions
                
                # Bullish divergence: Price down/sideways but Smart Money accumulating
                if price_change < 5 and sm_change > 1:
                    tech_score += 30
                # Strong bullish: Both price up and smart money in
                elif price_change > 0 and sm_change > 2:
                    tech_score += 25
                # Neutral
                elif -10 < price_change < 10:
                    tech_score += 10
            
            # RSI condition
            if 'RSI_14' in latest:
                if latest['RSI_14'] < 40:  # Oversold
                    tech_score += 15
                elif latest['RSI_14'] < 30:  # Very oversold
                    tech_score += 25
            
            # Volume confirmation
            if 'Volume' in recent_data.columns and 'Unusual Volume' in recent_data.columns:
                avg_volume = recent_data['Volume'].mean()
                if latest['Volume'] > avg_volume * 1.5:
                    tech_score += 10
                if latest.get('Unusual Volume', False):
                    tech_score += 5
            
            scores['technical'] = min(100, max(0, tech_score))
            
            # 3. FUNDAMENTAL & STRUCTURAL (30%)
            funda_score = 50
            
            # Free Float analysis (20-40% is optimal for potential squeeze)
            if 'Free Float' in latest:
                ff = latest['Free Float']
                if 20 <= ff <= 40:
                    funda_score += 30  # Perfect range
                elif ff < 20:
                    funda_score += 20  # Very concentrated (potential squeeze)
                elif ff < 10:
                    funda_score += 10  # Too concentrated
                elif ff > 60:
                    funda_score -= 10  # Too diluted
            
            # Ownership concentration trend
            if 'Ownership_Concentration' in recent_data.columns:
                if len(recent_data) > 10:
                    conc_start = recent_data.iloc[0]['Ownership_Concentration']
                    conc_end = latest['Ownership_Concentration']
                    if conc_end > conc_start + 5:  # Increasing concentration
                        funda_score += 15
            
            # Sector momentum (placeholder - would need sector data)
            funda_score += 10  # Base sector score
            
            scores['fundamental'] = min(100, max(0, funda_score))
            
            # 4. SENTIMENT & RISK (Adjustment factor, not additive)
            risk_adjustment = 1.0  # Default
            
            # High retail selling is good (contrarian)
            if 'Retail_Flow' in recent_data.columns:
                retail_total = recent_data['Retail_Flow'].sum()
                if retail_total < -1e9:  # Retail selling > 1B
                    risk_adjustment *= 1.1  # 10% bonus
                elif retail_total > 1e9:  # Retail buying > 1B
                    risk_adjustment *= 0.9  # 10% penalty
            
            # Market cap consideration (proxy via price and volume)
            if 'Value' in recent_data.columns:
                avg_daily_value = recent_data['Value'].mean()
                if avg_daily_value < 10e9:  # < 10B daily value (small-mid cap)
                    risk_adjustment *= 1.15  # 15% bonus for small caps
                elif avg_daily_value > 100e9:  # > 100B (large cap)
                    risk_adjustment *= 0.9  # 10% penalty for large caps
            
            # Calculate weighted total with risk adjustment
            weights = {'smart_money': 0.40, 'technical': 0.30, 'fundamental': 0.30}
            total_score = (
                scores['smart_money'] * weights['smart_money'] +
                scores['technical'] * weights['technical'] +
                scores['fundamental'] * weights['fundamental']
            ) * risk_adjustment
            
            # Ensure score between 0-100
            final_score = min(100, max(0, total_score))
            
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
                'latest_price': latest.get('Close', 0),
                'price_change_60d': ((latest.get('Close', 0) - recent_data.iloc[0].get('Close', 0)) / 
                                    recent_data.iloc[0].get('Close', 0) * 100) if recent_data.iloc[0].get('Close', 0) > 0 else 0,
                'sector': latest.get('Sector', 'N/A'),
                'free_float': latest.get('Free Float', 0),
                'smart_money_60d': recent_data['Smart_Money_Flow'].sum() if 'Smart_Money_Flow' in recent_data.columns else 0,
                'retail_60d': recent_data['Retail_Flow'].sum() if 'Retail_Flow' in recent_data.columns else 0,
                'institutional_net': latest.get('Institutional_Net', 0),
                'volume_avg_60d': recent_data['Volume'].mean() if 'Volume' in recent_data.columns else 0,
                'rsi': latest.get('RSI_14', 50)
            }
            
        except Exception as e:
            st.error(f"Error calculating score for {stock_code}: {e}")
            return {'total_score': 0, 'component_scores': {}, 'signal': 'ERROR', 'signal_color': 'neutral'}
    
    def find_top_gems(self, top_n=25, min_score=65, sector_filter=None):
        """Find top hidden gem candidates"""
        unique_stocks = self.df['Stock Code'].unique()
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_stocks = min(200, len(unique_stocks))  # Limit for performance
        
        for i, stock in enumerate(unique_stocks[:total_stocks]):
            status_text.text(f"🔍 Analyzing {stock}... ({i+1}/{total_stocks})")
            
            score_data = self.calculate_gem_score(stock)
            if score_data['total_score'] >= min_score:
                # Apply sector filter if specified
                if sector_filter and score_data['sector'] != sector_filter:
                    continue
                    
                results.append({
                    'Stock': stock,
                    'Gem Score': score_data['total_score'],
                    'Signal': score_data['signal'],
                    'Sector': score_data['sector'],
                    'Price': score_data['latest_price'],
                    'Price Chg 60D': score_data['price_change_60d'],
                    'Free Float %': score_data['free_float'],
                    'Smart Money 60D (B)': score_data['smart_money_60d'] / 1e9,
                    'Retail 60D (B)': score_data['retail_60d'] / 1e9,
                    'Institutional Net (B)': score_data['institutional_net'] / 1e9,
                    'Avg Volume 60D': score_data['volume_avg_60d'],
                    'RSI': score_data['rsi'],
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
        r=values + [values[0]],  # Close the loop
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(255, 215, 0, 0.3)',  # Gold color for gems
        line=dict(color='#FFD700', width=3)
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(color='#A3AED0')
            ),
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
    """Create timeline of ownership changes"""
    stock_data = df[df['Stock Code'] == stock_code].sort_values('Date')
    
    if stock_data.empty:
        return None
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'{stock_code} - Price & Smart Money Flow', 'Institutional vs Retail Flow'),
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4]
    )
    
    # Price line
    fig.add_trace(
        go.Scatter(x=stock_data['Date'], y=stock_data['Close'], 
                  name='Price', line=dict(color='#4318FF', width=2)),
        row=1, col=1
    )
    
    # Smart Money bars
    if 'Smart_Money_Flow' in stock_data.columns:
        colors = ['#05CD99' if x > 0 else '#EE5D50' for x in stock_data['Smart_Money_Flow']]
        fig.add_trace(
            go.Bar(x=stock_data['Date'], y=stock_data['Smart_Money_Flow'] / 1e9,
                  name='Smart Money (B)', marker_color=colors, opacity=0.7),
            row=1, col=1, secondary_y=True
        )
    
    # Institutional vs Retail flow
    if 'Institutional_Net' in stock_data.columns:
        fig.add_trace(
            go.Bar(x=stock_data['Date'], y=stock_data['Institutional_Net'] / 1e9,
                  name='Institutional Net (B)', marker_color='#4318FF'),
            row=2, col=1
        )
    
    if 'Retail_Flow' in stock_data.columns:
        fig.add_trace(
            go.Scatter(x=stock_data['Date'], y=stock_data['Retail_Flow'] / 1e9,
                      name='Retail Flow (B)', line=dict(color='#EE5D50', width=2)),
            row=2, col=1
        )
    
    fig.update_layout(
        height=600,
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
        )
    )
    
    fig.update_xaxes(showgrid=True, gridcolor='#E0E5F2', tickfont=dict(color='#A3AED0'))
    fig.update_yaxes(showgrid=True, gridcolor='#E0E5F2', tickfont=dict(color='#A3AED0'))
    
    return fig

def create_sector_heatmap(gems_df):
    """Create heatmap of gems by sector"""
    if gems_df.empty or 'Sector' not in gems_df.columns:
        return None
    
    sector_stats = gems_df.groupby('Sector').agg({
        'Stock': 'count',
        'Gem Score': 'mean',
        'Smart Money 60D (B)': 'sum'
    }).reset_index()
    
    sector_stats.columns = ['Sector', 'Count', 'Avg Score', 'Total Smart Money (B)']
    
    fig = px.treemap(
        sector_stats,
        path=['Sector'],
        values='Total Smart Money (B)',
        color='Avg Score',
        color_continuous_scale='RdYlGn',
        title='💎 Hidden Gems by Sector (Size = Smart Money, Color = Avg Score)',
        hover_data=['Count', 'Avg Score']
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#2B3674')
    )
    
    return fig

# ==============================================================================
# 🎨 MAIN DASHBOARD - 8 TABS VERSION
# ==============================================================================
def main():
    # HEADER
    st.markdown("""
    <div class="header-banner">
        <div class="header-title">🚀 HIDDEN GEM FINDER - KSEI & DAILY TX1Y MERGE</div>
        <div class="header-subtitle">Discover Undervalued Stocks with Institutional Accumulation • Multi-dimensional Analysis</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize data loader
    loader = DataLoader()
    
    if not loader.service:
        st.error("❌ Failed to initialize Google Drive service")
        return
    
    # Load data
    with st.spinner("🚀 Loading and analyzing merged dataset..."):
        df_merged = loader.load_merged_data()
    
    if df_merged.empty:
        st.error("Failed to load data. Please check data files.")
        return
    
    # Store in session for analyzer
    analyzer = HiddenGemAnalyzer(df_merged)
    st.session_state.analyzer = analyzer
    st.session_state.df_merged = df_merged
    
    # SIDEBAR
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=60)
        st.markdown("<h3 style='color:#2B3674;'>💎 Gem Finder</h3>", unsafe_allow_html=True)
        st.divider()
        
        # Quick stats
        st.markdown("##### 📊 Quick Stats")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric("Stocks", df_merged['Stock Code'].nunique())
        with col_s2:
            st.metric("Latest Date", df_merged['Date'].max().strftime('%d/%m'))
        
        st.divider()
        
        # Gem finder settings
        st.markdown("##### ⚙️ Gem Finder Settings")
        lookback_days = st.slider("Analysis Period (Days)", 30, 180, 60, 10)
        min_gem_score = st.slider("Minimum Gem Score", 50, 90, 65, 5)
        top_n_gems = st.slider("Top N Gems", 10, 50, 25, 5)
        
        # Sector filter
        sectors = ['All'] + sorted(df_merged['Sector'].unique().tolist())
        selected_sector = st.selectbox("Filter by Sector", sectors)
        
        st.divider()
        
        # Actions
        if st.button("🔄 Refresh All Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.divider()
        
        # Data info
        with st.expander("📁 Data Information"):
            st.write(f"**Merged Dataset:** {df_merged.shape[0]:,} rows × {df_merged.shape[1]:,} columns")
            st.write(f"**Date Range:** {df_merged['Date'].min().date()} to {df_merged['Date'].max().date()}")
            st.write(f"**Stocks:** {df_merged['Stock Code'].nunique():,}")
            st.write(f"**Latest Data:** {df_merged['Date'].max().strftime('%d %b %Y')}")
    
    # MAIN CONTENT - 8 TABS
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🏆 Top Gems", "📈 Stock Analyzer", "📊 Market Overview", 
        "🔄 Sector Rotation", "🧠 Smart Money Flow", "📉 Technical Scan",
        "🔍 Deep Dive", "📁 Data Explorer"
    ])
    
    # TAB 1: TOP GEMS
    with tab1:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">💎 Top Hidden Gem Candidates</div>', unsafe_allow_html=True)
        
        col_t1, col_t2 = st.columns([3, 1])
        with col_t1:
            st.markdown(f"**Period:** Last {lookback_days} days | **Min Score:** {min_gem_score} | **Sector:** {selected_sector}")
        with col_t2:
            if st.button("🔍 Find Gems", type="primary", use_container_width=True):
                st.session_state.find_gems_clicked = True
        
        if 'find_gems_clicked' in st.session_state and st.session_state.find_gems_clicked:
            sector_filter = None if selected_sector == 'All' else selected_sector
            
            with st.spinner(f"Finding hidden gems (min score: {min_gem_score})..."):
                gems_df = analyzer.find_top_gems(
                    top_n=top_n_gems, 
                    min_score=min_gem_score,
                    sector_filter=sector_filter
                )
            
            if not gems_df.empty:
                # Display summary
                col_sm1, col_sm2, col_sm3, col_sm4 = st.columns(4)
                with col_sm1:
                    st.metric("Total Gems", len(gems_df))
                with col_sm2:
                    st.metric("Avg Score", f"{gems_df['Gem Score'].mean():.1f}")
                with col_sm3:
                    st.metric("Avg Smart Money", f"{gems_df['Smart Money 60D (B)'].sum():.1f}B")
                with col_sm4:
                    st.metric("Top Sector", gems_df['Sector'].mode().iloc[0] if not gems_df.empty else "N/A")
                
                # Sector heatmap
                st.markdown("#### 🗺️ Sector Distribution")
                heatmap = create_sector_heatmap(gems_df)
                if heatmap:
                    st.plotly_chart(heatmap, use_container_width=True)
                
                # Gems table
                st.markdown("#### 🏆 Gem Candidates")
                
                # Format the dataframe for display
                display_cols = ['Stock', 'Gem Score', 'Signal', 'Sector', 'Price', 
                              'Price Chg 60D', 'Free Float %', 'Smart Money 60D (B)', 
                              'RSI']
                
                display_df = gems_df[display_cols].copy()
                
                # Apply styling
                def color_signal(val):
                    if 'BUY' in val or 'ACCUMULATE' in val:
                        return 'color: #065F46; font-weight: bold;'
                    elif 'WATCH' in val:
                        return 'color: #D97706; font-weight: bold;'
                    else:
                        return 'color: #6B7280;'
                
                st.dataframe(
                    display_df.style
                    .format({
                        'Price': '{:,.0f}',
                        'Price Chg 60D': '{:.1f}%',
                        'Free Float %': '{:.1f}%',
                        'Smart Money 60D (B)': '{:.2f}',
                        'RSI': '{:.1f}',
                        'Gem Score': '{:.1f}'
                    })
                    .applymap(color_signal, subset=['Signal'])
                    .background_gradient(
                        subset=['Gem Score'], 
                        cmap='RdYlGn', 
                        vmin=65, 
                        vmax=100
                    )
                    .bar(subset=['Smart Money 60D (B)'], color='#05CD99')
                    .bar(subset=['Price Chg 60D'], color='#4318FF', align='zero'),
                    use_container_width=True,
                    hide_index=True,
                    height=600
                )
                
                # Top gem analysis
                if len(gems_df) > 0:
                    st.markdown("#### 🎯 Top Gem Analysis")
                    top_gem = gems_df.iloc[0]['Stock']
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
                    
                    # Detailed metrics
                    with st.expander(f"📊 Detailed Analysis: {top_gem}"):
                        col_d1, col_d2, col_d3, col_d4 = st.columns(4)
                        with col_d1:
                            st.metric("Gem Score", f"{score_data['total_score']:.1f}/100")
                        with col_d2:
                            st.metric("Signal", score_data['signal'])
                        with col_d3:
                            st.metric("Smart Money 60D", f"Rp {score_data['smart_money_60d']/1e9:.1f}B")
                        with col_d4:
                            st.metric("Retail Flow 60D", f"Rp {score_data['retail_60d']/1e9:.1f}B")
                        
                        st.write("**Component Scores:**")
                        comp_df = pd.DataFrame.from_dict(
                            score_data['component_scores'], 
                            orient='index', 
                            columns=['Score']
                        )
                        comp_df['Weight'] = [0.40, 0.30, 0.30]
                        comp_df['Weighted'] = comp_df['Score'] * comp_df['Weight']
                        st.dataframe(
                            comp_df.style.format({'Score': '{:.1f}', 'Weight': '{:.2f}', 'Weighted': '{:.2f}'}),
                            use_container_width=True
                        )
            else:
                st.info("No hidden gems found with the current criteria. Try lowering the minimum score or changing the sector filter.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # TAB 2: STOCK ANALYZER
    with tab2:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🔍 Individual Stock Analysis</div>', unsafe_allow_html=True)
        
        # Stock selector
        available_stocks = sorted(df_merged['Stock Code'].unique())
        selected_stock = st.selectbox(
            "Select Stock", 
            available_stocks,
            index=available_stocks.index('BBRI') if 'BBRI' in available_stocks else 0
        )
        
        if selected_stock:
            # Calculate score
            score_data = analyzer.calculate_gem_score(selected_stock, lookback_days)
            
            # Display header metrics
            col_h1, col_h2, col_h3, col_h4 = st.columns(4)
            with col_h1:
                st.metric("Gem Score", f"{score_data['total_score']:.1f}/100", 
                         delta=score_data['signal'], delta_color="off")
            with col_h2:
                st.metric("Price", f"Rp {score_data['latest_price']:,.0f}",
                         delta=f"{score_data['price_change_60d']:.1f}% 60D")
            with col_h3:
                st.metric("Smart Money 60D", f"Rp {score_data['smart_money_60d']/1e9:.1f}B",
                         delta=f"Rp {score_data['retail_60d']/1e9:.1f}B Retail")
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
            
            # Detailed analysis
            with st.expander("📈 Technical & Fundamental Details"):
                col_det1, col_det2, col_det3 = st.columns(3)
                
                with col_det1:
                    st.markdown("##### 🧠 Smart Money Analysis")
                    st.write(f"**60D Total:** Rp {score_data['smart_money_60d']/1e9:.2f}B")
                    st.write(f"**Institutional Net:** Rp {score_data['institutional_net']/1e9:.2f}B")
                    st.write(f"**Retail Flow:** Rp {score_data['retail_60d']/1e9:.2f}B")
                
                with col_det2:
                    st.markdown("##### 📊 Technical Indicators")
                    st.write(f"**RSI (14):** {score_data['rsi']:.1f}")
                    st.write(f"**60D Price Change:** {score_data['price_change_60d']:.1f}%")
                    st.write(f"**Avg Daily Volume:** {score_data['volume_avg_60d']/1e6:.1f}M")
                
                with col_det3:
                    st.markdown("##### 🏢 Fundamental")
                    st.write(f"**Sector:** {score_data['sector']}")
                    st.write(f"**Free Float:** {score_data['free_float']:.1f}%")
                    st.write(f"**Ownership Concentration:** {100 - score_data['free_float']:.1f}%")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # TAB 3: MARKET OVERVIEW
    with tab3:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📊 Market Overview</div>', unsafe_allow_html=True)
        
        # Date selector
        latest_dates = sorted(df_merged['Date'].unique(), reverse=True)[:30]
        selected_date = st.selectbox(
            "Select Date for Overview",
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
                    unusual_count = daily_data['Unusual Volume'].sum() if 'Unusual Volume' in daily_data.columns else 0
                    st.metric("Unusual Volume", unusual_count)
                
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
                        }).background_gradient(
                            subset=['Change %'], 
                            cmap='Greens'
                        ),
                        use_container_width=True,
                        hide_index=True
                    )
                
                with col_tab2:
                    st.markdown("##### 💰 Top Value Stocks")
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
                
                # Sector performance
                st.markdown("##### 🏭 Sector Performance")
                if 'Sector' in daily_data.columns and 'Change %' in daily_data.columns:
                    sector_perf = daily_data.groupby('Sector').agg({
                        'Change %': 'mean',
                        'Value': 'sum',
                        'Stock Code': 'count'
                    }).round(2).reset_index()
                    
                    sector_perf.columns = ['Sector', 'Avg Change %', 'Total Value', 'Stock Count']
                    sector_perf['Total Value_T'] = sector_perf['Total Value'] / 1e12
                    
                    fig_sector = px.bar(
                        sector_perf.sort_values('Avg Change %', ascending=False),
                        x='Sector',
                        y='Avg Change %',
                        color='Total Value_T',
                        color_continuous_scale='RdYlGn',
                        title=f"Sector Performance - {selected_date.strftime('%d %b %Y')}",
                        hover_data=['Stock Count', 'Total Value_T']
                    )
                    st.plotly_chart(fig_sector, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # TAB 4: SECTOR ROTATION
    with tab4:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🔄 Sector Rotation Analysis</div>', unsafe_allow_html=True)
        
        # Date range selector
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
            
            if not period_data.empty:
                # Sector-wise smart money flow
                if 'Smart_Money_Flow' in period_data.columns:
                    sector_flow = period_data.groupby('Sector').agg({
                        'Smart_Money_Flow': 'sum',
                        'Retail_Flow': 'sum',
                        'Stock Code': 'nunique'
                    }).reset_index()
                    
                    sector_flow.columns = ['Sector', 'Smart Money Flow', 'Retail Flow', 'Stock Count']
                    sector_flow['Net Institutional'] = sector_flow['Smart Money Flow'] - sector_flow['Retail Flow']
                    sector_flow['Smart Money Flow_B'] = sector_flow['Smart Money Flow'] / 1e9
                    sector_flow['Net Institutional_B'] = sector_flow['Net Institutional'] / 1e9
                    
                    # Display charts
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
                            title="Top Sector Inflows (Net Institutional, B)",
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
                            title="Top Sector Outflows (Net Institutional, B)",
                            hover_data=['Stock Count']
                        )
                        st.plotly_chart(fig_outflows, use_container_width=True)
                    
                    # Sector heatmap
                    st.markdown("##### 🔥 Sector Flow Matrix")
                    st.dataframe(
                        sector_flow[['Sector', 'Smart Money Flow_B', 'Retail Flow', 'Net Institutional_B', 'Stock Count']]
                        .sort_values('Net Institutional_B', ascending=False)
                        .style.format({
                            'Smart Money Flow_B': '{:.1f}B',
                            'Retail Flow': '{:.0f}',
                            'Net Institutional_B': '{:.1f}B'
                        }).background_gradient(
                            subset=['Net Institutional_B'], 
                            cmap='RdYlGn'
                        ),
                        use_container_width=True,
                        hide_index=True
                    )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # TAB 5-8: PLACEHOLDERS (akan kita kembangkan nanti)
    with tab5:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🧠 Smart Money Flow Analysis</div>', unsafe_allow_html=True)
        st.info("""
        **Coming Soon Features:**
        - Real-time smart money tracking
        - Accumulation/distribution patterns
        - Cross-market comparisons
        - Historical flow analysis
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab6:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📉 Technical Scan</div>', unsafe_allow_html=True)
        st.info("""
        **Coming Soon Features:**
        - Multi-timeframe analysis
        - Pattern recognition
        - Support/resistance levels
        - Volume profile analysis
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab7:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🔍 Deep Dive Analysis</div>', unsafe_allow_html=True)
        st.info("""
        **Coming Soon Features:**
        - Peer comparison
        - Valuation metrics
        - Risk assessment
        - Scenario analysis
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab8:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📁 Data Explorer</div>', unsafe_allow_html=True)
        
        # Quick data exploration
        st.dataframe(
            df_merged.describe().T.style.format('{:,.2f}'),
            use_container_width=True
        )
        
        with st.expander("🔍 View Sample Data"):
            st.dataframe(df_merged.head(50), use_container_width=True)
        
        with st.expander("📊 Column Information"):
            col_info = pd.DataFrame({
                'Column': df_merged.columns,
                'Type': df_merged.dtypes.astype(str),
                'Non-Null': df_merged.count().values,
                'Null %': (df_merged.isnull().sum().values / len(df_merged) * 100).round(1)
            })
            st.dataframe(col_info, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # FOOTER
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #A3AED0; font-size: 14px;'>"
        "💎 HIDDEN GEM FINDER v1.0 • KSEI + Daily TX1Y Merge • "
        f"Data as of {df_merged['Date'].max().strftime('%d %b %Y')} • "
        f"Total Stocks: {df_merged['Stock Code'].nunique():,}"
        "</div>",
        unsafe_allow_html=True
    )

# ==============================================================================
# 🚀 RUN APP
# ==============================================================================
if __name__ == "__main__":
    main()
