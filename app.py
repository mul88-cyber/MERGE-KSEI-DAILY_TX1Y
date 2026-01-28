# ==============================================================================
# 🚀 HIDDEN GEM FINDER v3.0 - ENTERPRISE EDITION (OPTIMIZED)
# ==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import time
import logging
from datetime import datetime, timedelta
from functools import wraps
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
    CACHE_TTL = 3600
    
    # Reduce for testing - bisa dinaikkan kalau sudah work
    MAX_STOCKS_ANALYZED = 200
    
    # Scoring weights
    SCORE_WEIGHTS = {
        'smart_money': 0.40,
        'technical': 0.30,
        'fundamental': 0.15,
        'volatility': 0.15
    }
    
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
    page_title="💎 HIDDEN GEM FINDER v3.0",
    layout="wide",
    page_icon="💎",
    initial_sidebar_state="expanded"
)

# --- CSS ---
st.markdown("""
<style>
    .stApp {
        background-color: #F4F7FE;
        color: #2B3674;
        font-family: 'DM Sans', sans-serif;
    }
    
    .header-gradient {
        background: linear-gradient(135deg, #4318FF 0%, #868CFF 100%);
        border-radius: 20px;
        padding: 30px 40px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0px 18px 40px rgba(112, 144, 176, 0.2);
    }
    
    .header-title { 
        font-size: 2.5rem; 
        font-weight: 800; 
        margin-bottom: 10px;
    }
    
    .header-subtitle { 
        font-size: 1.1rem; 
        font-weight: 500; 
        opacity: 0.9;
    }
    
    .css-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 28px;
        border: 1px solid rgba(67, 24, 255, 0.1);
        box-shadow: 0px 18px 40px rgba(112, 144, 176, 0.12);
        margin-bottom: 24px;
    }
    
    .card-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2B3674;
        margin-bottom: 24px;
        border-bottom: 2px solid #4318FF;
        padding-bottom: 12px;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8F9FF 100%);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(67, 24, 255, 0.1);
        height: 100%;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: transparent;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        padding: 0 24px;
        background-color: rgba(255, 255, 255, 0.7);
        border-radius: 10px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #4318FF 0%, #868CFF 100%);
        color: white !important;
    }
    
    div.stButton > button {
        background: linear-gradient(90deg, #4318FF 0%, #868CFF 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        width: 100%;
    }
    
    .signal-buy { 
        background: linear-gradient(90deg, #05CD99 0%, #00B894 100%);
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
    
    .signal-sell { 
        background: linear-gradient(90deg, #EE5D50 0%, #FF6B6B 100%);
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 🛠️ UTILITY FUNCTIONS
# ==============================================================================
class PerformanceMonitor:
    """Performance monitoring"""
    
    @staticmethod
    def time_execution(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            if 'performance_metrics' not in st.session_state:
                st.session_state.performance_metrics = {}
            
            st.session_state.performance_metrics[func.__name__] = {
                'elapsed': elapsed,
                'timestamp': datetime.now().isoformat()
            }
            
            return result
        return wrapper

class ErrorHandler:
    """Error handling"""
    
    def __init__(self):
        self.logger = logging.getLogger('GemFinder')
        self.logger.setLevel(logging.ERROR)
    
    def safe_execute(self, func, *args, fallback_value=None, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Error in {func.__name__}: {str(e)}")
            
            if fallback_value is not None:
                return fallback_value
            
            return None

# ==============================================================================
# 📦 OPTIMIZED DATA LOADER
# ==============================================================================
class OptimizedDataLoader:
    """Optimized data loader dengan cache yang benar"""
    
    def __init__(self):
        self.service = None
        self.error_handler = ErrorHandler()
        self._init_gdrive()
    
    def _init_gdrive(self):
        """Initialize Google Drive"""
        try:
            if "gcp_service_account" not in st.secrets:
                st.error("❌ 'gcp_service_account' tidak ditemukan")
                return False
            
            creds_data = st.secrets["gcp_service_account"]
            if hasattr(creds_data, "to_dict"):
                creds_json = creds_data.to_dict()
            else:
                creds_json = dict(creds_data)
            
            # Fix private key
            if "private_key" in creds_json:
                pk = str(creds_json["private_key"])
                creds_json["private_key"] = pk.replace("\\n", "\n")
            
            # Use read-only scope
            SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
            creds = Credentials.from_service_account_info(creds_json)
            scoped_creds = creds.with_scopes(SCOPES)
            
            self.service = build('drive', 'v3', credentials=scoped_creds, cache_discovery=False)
            return True
            
        except Exception as e:
            st.error(f"❌ Google Drive Error: {str(e)}")
            return False
    
    def _download_file(self, file_name):
        """Download file dari Google Drive"""
        if not self.service:
            return None, "Service not initialized"
        
        try:
            # Cari file
            query = f"'{Config.FOLDER_ID}' in parents and name='{file_name}' and trashed=false"
            results = self.service.files().list(q=query, fields="files(id, name)", pageSize=1).execute()
            items = results.get('files', [])
            
            if not items:
                return None, f"File '{file_name}' not found"
            
            # Download
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
    @st.cache_data(ttl=3600, show_spinner=False)
    def load_all_data(_self):
        """Load SEMUA data dalam SATU fungsi dengan cache"""
        print("🚀 Starting data load process...")
        
        # 1. Load KSEI Data
        print("📥 Downloading KSEI data...")
        fh_ksei, error_ksei = _self._download_file(Config.FILE_KSEI)
        if error_ksei:
            print(f"❌ KSEI Error: {error_ksei}")
            return pd.DataFrame()
        
        try:
            df_ksei = pd.read_csv(fh_ksei, dtype=str)
            print(f"✅ KSEI loaded: {len(df_ksei)} rows, {len(df_ksei.columns)} columns")
        except Exception as e:
            print(f"❌ KSEI Parse Error: {e}")
            return pd.DataFrame()
        
        # 2. Load Historical Data
        print("📊 Downloading Historical data...")
        fh_hist, error_hist = _self._download_file(Config.FILE_HIST)
        if error_hist:
            print(f"❌ Historical Error: {error_hist}")
            return pd.DataFrame()
        
        try:
            df_hist = pd.read_csv(fh_hist, dtype=str)
            print(f"✅ Historical loaded: {len(df_hist)} rows, {len(df_hist.columns)} columns")
        except Exception as e:
            print(f"❌ Historical Parse Error: {e}")
            return pd.DataFrame()
        
        # 3. Process KSEI Data
        print("🔄 Processing KSEI data...")
        df_ksei = _self._process_ksei_data(df_ksei)
        if df_ksei.empty:
            print("❌ KSEI processing failed")
            return pd.DataFrame()
        
        # 4. Process Historical Data
        print("🔄 Processing Historical data...")
        df_hist = _self._process_historical_data(df_hist)
        if df_hist.empty:
            print("❌ Historical processing failed")
            return pd.DataFrame()
        
        # 5. Merge Data
        print("🔗 Merging datasets...")
        df_merged = _self._merge_data(df_ksei, df_hist)
        
        print(f"✅ Data load complete: {len(df_merged)} rows")
        return df_merged
    
    def _process_ksei_data(self, df):
        """Process KSEI dataframe"""
        if df.empty:
            return df
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Handle Date
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df[df['Date'].notna()].copy()
        else:
            print("❌ No 'Date' column in KSEI data")
            return pd.DataFrame()
        
        # Filter recent years
        df = df[df['Date'].dt.year >= 2023].copy()
        
        # Convert numeric columns
        numeric_cols = []
        for base_col in Config.OWNERSHIP_COLS:
            for suffix in ['', '_chg', '_chg_Rp']:
                col = f"{base_col}{suffix}"
                if col in df.columns:
                    numeric_cols.append(col)
        
        additional_cols = ['Price', 'Free Float', 'Total_Local', 'Total_Foreign']
        numeric_cols.extend([col for col in additional_cols if col in df.columns])
        
        for col in numeric_cols:
            if col in df.columns:
                try:
                    # Remove commas and convert to numeric
                    df[col] = pd.to_numeric(
                        df[col].astype(str).str.replace(',', '', regex=False),
                        errors='coerce'
                    ).fillna(0)
                except:
                    df[col] = 0
        
        # Calculate Smart Money Flow (include Local CP)
        smart_money_cols = [
            'Foreign IS_chg_Rp', 'Foreign IB_chg_Rp', 'Foreign PF_chg_Rp',
            'Local IS_chg_Rp', 'Local PF_chg_Rp', 'Local MF_chg_Rp', 
            'Local IB_chg_Rp', 'Local CP_chg_Rp'
        ]
        
        valid_sm_cols = [c for c in smart_money_cols if c in df.columns]
        df['Smart_Money_Flow'] = df[valid_sm_cols].sum(axis=1) if valid_sm_cols else 0
        
        # Retail Flow
        retail_cols = [c for c in ['Local ID_chg_Rp'] if c in df.columns]
        df['Retail_Flow'] = df[retail_cols].sum(axis=1) if retail_cols else 0
        
        # Institutional Net
        df['Institutional_Net'] = df['Smart_Money_Flow'] - df['Retail_Flow']
        
        # Ensure Stock Code column
        if 'Code' in df.columns:
            df['Stock Code'] = df['Code']
        elif 'Stock Code' not in df.columns:
            print("❌ No stock identifier column found")
            return pd.DataFrame()
        
        # Ownership Concentration
        if 'Free Float' in df.columns:
            df['Ownership_Concentration'] = 100 - df['Free Float']
        
        return df
    
    def _process_historical_data(self, df):
        """Process historical dataframe"""
        if df.empty:
            return df
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Handle Date
        date_col = 'Last Trading Date' if 'Last Trading Date' in df.columns else 'Date'
        if date_col in df.columns:
            df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
            df = df[df['Date'].notna()].copy()
        else:
            print("❌ No date column in Historical data")
            return pd.DataFrame()
        
        # Numeric columns to process
        numeric_cols = [
            'High', 'Low', 'Close', 'Volume', 'Value', 'Foreign Buy', 
            'Foreign Sell', 'Bid Volume', 'Offer Volume', 'Previous', 
            'Change', 'Open Price', 'Change %', 'Typical Price', 
            'Net Foreign Flow', 'Listed Shares'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                try:
                    # Clean string: remove commas, spaces, Rp, %
                    cleaned = (
                        df[col]
                        .astype(str)
                        .str.strip()
                        .str.replace(r'[,\sRp\%]', '', regex=True)
                    )
                    df[col] = pd.to_numeric(cleaned, errors='coerce').fillna(0)
                except:
                    df[col] = 0
        
        # Calculate NFF in Rupiah
        if 'Net Foreign Flow' in df.columns:
            if 'Typical Price' in df.columns:
                df['NFF_Rp'] = df['Net Foreign Flow'] * df['Typical Price']
            elif 'Close' in df.columns:
                df['NFF_Rp'] = df['Net Foreign Flow'] * df['Close']
        
        # Ensure Stock Code column
        if 'Stock Code' not in df.columns:
            if 'Stock' in df.columns:
                df['Stock Code'] = df['Stock']
            elif 'Code' in df.columns:
                df['Stock Code'] = df['Code']
            else:
                print("❌ No stock identifier in Historical data")
                return pd.DataFrame()
        
        # Sector
        if 'Sector' in df.columns:
            df['Sector'] = df['Sector'].astype(str).str.strip().fillna('Others')
        else:
            df['Sector'] = 'Others'
        
        return df
    
    def _merge_data(self, df_ksei, df_hist):
        """Merge KSEI and Historical data"""
        if df_ksei.empty or df_hist.empty:
            return pd.DataFrame()
        
        # Sort both dataframes
        df_ksei = df_ksei.sort_values(['Stock Code', 'Date']).reset_index(drop=True)
        df_hist = df_hist.sort_values(['Stock Code', 'Date']).reset_index(drop=True)
        
        # KSEI columns to merge
        ksei_cols = ['Date', 'Stock Code', 'Smart_Money_Flow', 'Institutional_Net', 
                    'Retail_Flow', 'Free Float', 'Ownership_Concentration']
        ksei_cols = [c for c in ksei_cols if c in df_ksei.columns]
        
        # Create month-end indicator for KSEI data
        df_ksei['is_month_end'] = df_ksei['Date'].dt.is_month_end
        
        # For each stock, forward fill KSEI data to daily
        merged_list = []
        unique_stocks = df_hist['Stock Code'].unique()[:100]  # Limit for testing
        
        print(f"🔄 Merging {len(unique_stocks)} stocks...")
        
        for i, stock in enumerate(unique_stocks):
            if i % 20 == 0:
                print(f"   Processing stock {i+1}/{len(unique_stocks)}: {stock}")
            
            hist_stock = df_hist[df_hist['Stock Code'] == stock].copy()
            ksei_stock = df_ksei[df_ksei['Stock Code'] == stock].copy()
            
            if hist_stock.empty or ksei_stock.empty:
                continue
            
            # Merge using forward fill
            merged_stock = hist_stock.copy()
            
            for col in ksei_cols:
                if col in ksei_stock.columns:
                    # Create series for forward filling
                    ksei_dates = ksei_stock['Date'].values
                    ksei_values = ksei_stock[col].values
                    
                    # Forward fill to historical dates
                    merged_stock[col] = np.interp(
                        merged_stock['Date'].astype(np.int64) // 10**9,
                        ksei_dates.astype(np.int64) // 10**9,
                        ksei_values,
                        left=np.nan,
                        right=np.nan
                    )
            
            merged_list.append(merged_stock)
        
        if not merged_list:
            return pd.DataFrame()
        
        # Combine all stocks
        merged = pd.concat(merged_list, ignore_index=True)
        
        # Fill remaining NaN with forward fill per stock
        for col in ksei_cols:
            if col in merged.columns:
                merged[col] = merged.groupby('Stock Code')[col].ffill()
        
        # Clean up
        merged = merged.dropna(subset=['Close', 'Smart_Money_Flow'], how='all')
        
        return merged

# ==============================================================================
# 🎯 GEM ANALYZER
# ==============================================================================
class GemAnalyzer:
    """Gem analyzer dengan caching"""
    
    def __init__(self, df_merged):
        self.df = df_merged.copy()
        self.latest_date = self.df['Date'].max() if not df_merged.empty else None
        self.error_handler = ErrorHandler()
    
    @st.cache_data(ttl=1800, show_spinner=False)
    def calculate_gem_score_cached(_self, stock_code, lookback_days=90):
        """Cached version of gem score calculation"""
        return _self._calculate_gem_score(stock_code, lookback_days)
    
    def _calculate_gem_score(self, stock_code, lookback_days=90):
        """Calculate gem score untuk satu saham"""
        stock_data = self.df[self.df['Stock Code'] == stock_code].sort_values('Date')
        if stock_data.empty:
            return {'total_score': 0, 'signal': 'NO DATA', 'components': {}}
        
        cutoff_date = self.latest_date - timedelta(days=lookback_days)
        recent_data = stock_data[stock_data['Date'] >= cutoff_date]
        
        if recent_data.empty or len(recent_data) < 20:
            return {'total_score': 0, 'signal': 'INSUFFICIENT DATA', 'components': {}}
        
        latest = recent_data.iloc[-1]
        
        # Calculate component scores
        components = {}
        
        # 1. Smart Money Score (40%)
        sm_score = self._calculate_smart_money_score(recent_data)
        components['smart_money'] = sm_score
        
        # 2. Technical Score (30%)
        tech_score = self._calculate_technical_score(recent_data, latest)
        components['technical'] = tech_score
        
        # 3. Fundamental Score (15%)
        fund_score = self._calculate_fundamental_score(latest)
        components['fundamental'] = fund_score
        
        # 4. Volatility Score (15%)
        vol_score = self._calculate_volatility_score(recent_data)
        components['volatility'] = vol_score
        
        # Calculate total score
        total_score = (
            components['smart_money'] * Config.SCORE_WEIGHTS['smart_money'] +
            components['technical'] * Config.SCORE_WEIGHTS['technical'] +
            components['fundamental'] * Config.SCORE_WEIGHTS['fundamental'] +
            components['volatility'] * Config.SCORE_WEIGHTS['volatility']
        )
        
        # Apply market regime multiplier
        market_trend = self._analyze_market_trend()
        regime_multiplier = Config.REGIME_MULTIPLIERS.get(market_trend, 1.0)
        total_score *= regime_multiplier
        
        # Determine signal
        signal = self._determine_signal(total_score, latest)
        
        # Get monthly data
        monthly_data = self._get_monthly_data(recent_data)
        
        return {
            'total_score': round(total_score, 1),
            'components': components,
            'signal': signal,
            'latest_price': latest.get('Close', 0),
            'price_change': self._calculate_price_change(recent_data),
            'sector': latest.get('Sector', 'N/A'),
            'free_float': latest.get('Free Float', 0),
            'smart_money_total': monthly_data['Smart_Money_Flow'].sum() if not monthly_data.empty else 0,
            'positive_months': (monthly_data['Smart_Money_Flow'] > 0).sum() if not monthly_data.empty else 0,
            'total_months': len(monthly_data),
            'rsi': latest.get('RSI_14', 50) if 'RSI_14' in latest else 50,
            'market_trend': market_trend
        }
    
    def _calculate_smart_money_score(self, recent_data):
        """Calculate smart money score"""
        monthly_data = self._get_monthly_data(recent_data)
        
        if monthly_data.empty or 'Smart_Money_Flow' not in monthly_data.columns:
            return 50
        
        sm_flow = monthly_data['Smart_Money_Flow']
        total_flow = sm_flow.sum()
        positive_months = (sm_flow > 0).sum()
        total_months = len(monthly_data)
        
        # Amount score (0-30)
        if abs(total_flow) > 10e9:  # > 10B
            amount_score = 30
        elif abs(total_flow) > 5e9:  # > 5B
            amount_score = 25
        elif abs(total_flow) > 2e9:  # > 2B
            amount_score = 20
        elif abs(total_flow) > 1e9:  # > 1B
            amount_score = 15
        elif abs(total_flow) > 0.5e9:  # > 500M
            amount_score = 10
        else:
            amount_score = 5
        
        # Consistency score (0-25)
        if total_months > 0:
            consistency_ratio = positive_months / total_months
            if consistency_ratio >= 0.8:
                consistency_score = 25
            elif consistency_ratio >= 0.6:
                consistency_score = 20
            elif consistency_ratio >= 0.4:
                consistency_score = 15
            elif consistency_ratio >= 0.2:
                consistency_score = 10
            else:
                consistency_score = 5
        else:
            consistency_score = 0
        
        # Trend score (0-20)
        if len(sm_flow) >= 3:
            # Calculate trend
            x = np.arange(len(sm_flow))
            trend = np.polyfit(x, sm_flow.values, 1)[0]
            
            if trend > 1e9:  # Strong upward trend
                trend_score = 20
            elif trend > 0.5e9:  # Moderate upward
                trend_score = 15
            elif trend > 0:  # Slight upward
                trend_score = 10
            elif trend > -0.5e9:  # Slight downward
                trend_score = 5
            else:  # Strong downward
                trend_score = 0
        else:
            trend_score = 10
        
        # Divergence bonus (0-15)
        price_change = self._calculate_price_change(recent_data)
        if total_flow > 2e9 and price_change < 10:  # Accumulation without price movement
            divergence_bonus = 15
        elif total_flow > 0 and price_change < 5:
            divergence_bonus = 10
        elif total_flow > 0 and price_change < 15:
            divergence_bonus = 5
        else:
            divergence_bonus = 0
        
        total = amount_score + consistency_score + trend_score + divergence_bonus + 10  # Base 10
        return min(100, total)
    
    def _calculate_technical_score(self, recent_data, latest):
        """Calculate technical score"""
        score = 50  # Base score
        
        # Price momentum (0-20)
        price_change = self._calculate_price_change(recent_data)
        if -10 <= price_change <= 25:  # Ideal range
            score += 20
        elif -15 <= price_change <= 30:  # Acceptable range
            score += 15
        elif -20 <= price_change <= 35:  # Okay range
            score += 10
        elif -25 <= price_change <= 40:  # Borderline
            score += 5
        
        # RSI (0-15)
        if 'RSI_14' in latest:
            rsi = latest['RSI_14']
            if 30 <= rsi <= 40:  # Oversold bounce potential
                score += 15
            elif 40 < rsi < 60:  # Neutral
                score += 10
            elif 60 <= rsi <= 70:  # Overbought
                score += 5
            elif rsi < 30:  # Very oversold
                score += 10
            else:  # Very overbought
                score += 0
        
        # Volume trend (0-10)
        if 'Volume' in recent_data.columns and len(recent_data) >= 10:
            avg_volume = recent_data['Volume'].mean()
            recent_avg = recent_data['Volume'].tail(5).mean()
            if avg_volume > 0:
                volume_ratio = recent_avg / avg_volume
                if volume_ratio > 1.5:
                    score += 10
                elif volume_ratio > 1.2:
                    score += 7
                elif volume_ratio > 0.8:
                    score += 5
                else:
                    score += 3
        
        # Moving average (0-10)
        if 'Close' in recent_data.columns and len(recent_data) >= 20:
            prices = recent_data['Close'].values
            ma20 = np.mean(prices[-20:]) if len(prices) >= 20 else prices[-1]
            current_price = prices[-1]
            
            if current_price > ma20 * 1.05:
                score += 10
            elif current_price > ma20:
                score += 7
            elif current_price > ma20 * 0.95:
                score += 5
            else:
                score += 3
        
        return min(100, score)
    
    def _calculate_fundamental_score(self, latest):
        """Calculate fundamental score"""
        score = 50  # Base score
        
        # Free Float (0-20)
        if 'Free Float' in latest:
            ff = latest['Free Float']
            if 20 <= ff <= 40:  # Ideal range
                score += 20
            elif 15 <= ff <= 50:  # Good range
                score += 15
            elif 10 <= ff <= 60:  # Acceptable
                score += 10
            elif ff < 10 or ff > 60:  # Problematic
                score += 5
        
        # Liquidity (0-15)
        if 'Value' in latest:
            daily_value = latest['Value']
            if daily_value > 50e9:  # > 50B
                score += 15
            elif daily_value > 20e9:  # > 20B
                score += 12
            elif daily_value > 10e9:  # > 10B
                score += 10
            elif daily_value > 5e9:  # > 5B
                score += 7
            elif daily_value > 1e9:  # > 1B
                score += 5
        
        # Market Cap (0-10)
        if 'Close' in latest and 'Listed Shares' in latest:
            market_cap = latest['Close'] * latest['Listed Shares']
            market_cap_t = market_cap / 1e12  # Trillion
            
            if market_cap_t < 1:  # Small cap - more potential
                score += 10
            elif market_cap_t < 5:  # Mid cap
                score += 7
            elif market_cap_t < 20:  # Large cap
                score += 5
            else:  # Mega cap
                score += 3
        
        # Sector momentum (0-5)
        score += 5  # Base sector score
        
        return min(100, score)
    
    def _calculate_volatility_score(self, recent_data):
        """Calculate volatility score (lower volatility = higher score)"""
        if 'Close' not in recent_data.columns or len(recent_data) < 20:
            return 50
        
        returns = recent_data['Close'].pct_change().dropna()
        if len(returns) < 10:
            return 50
        
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        if volatility < 0.25:  # Very low volatility
            return 90
        elif volatility < 0.35:  # Low volatility
            return 80
        elif volatility < 0.50:  # Moderate volatility
            return 70
        elif volatility < 0.75:  # High volatility
            return 60
        elif volatility < 1.00:  # Very high volatility
            return 40
        else:  # Extreme volatility
            return 20
    
    def _calculate_price_change(self, recent_data):
        """Calculate price change percentage"""
        if len(recent_data) < 2 or 'Close' not in recent_data.columns:
            return 0
        
        start_price = recent_data.iloc[0]['Close']
        end_price = recent_data.iloc[-1]['Close']
        
        if start_price > 0:
            return ((end_price - start_price) / start_price) * 100
        return 0
    
    def _get_monthly_data(self, stock_data):
        """Get monthly KSEI data"""
        if stock_data.empty:
            return pd.DataFrame()
        
        # Get month-end data
        month_ends = stock_data[stock_data['Date'].dt.is_month_end]
        if not month_ends.empty:
            return month_ends
        
        # If no month-end, take last day of each month
        stock_data['YearMonth'] = stock_data['Date'].dt.to_period('M')
        monthly = stock_data.loc[stock_data.groupby('YearMonth')['Date'].idxmax()]
        
        return monthly
    
    def _analyze_market_trend(self):
        """Analyze overall market trend"""
        # Simple implementation - bisa diimprove
        index_stocks = ['BBCA', 'BBRI', 'BMRI', 'TLKM', 'ASII']
        index_data = self.df[self.df['Stock Code'].isin(index_stocks)]
        
        if index_data.empty:
            return 'neutral'
        
        latest_date = self.df['Date'].max()
        latest_index = index_data[index_data['Date'] == latest_date]
        
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
    
    def _determine_signal(self, score, latest):
        """Determine buy/sell signal"""
        # Check liquidity
        avg_value = latest.get('Value', 0)
        
        if avg_value < 1e9:  # < 1B daily value
            return "⚠️ LOW LIQUIDITY"
        
        if score >= 85:
            return "💎 STRONG BUY"
        elif score >= 75:
            return "🔥 BUY"
        elif score >= 65:
            return "👀 WATCH"
        elif score >= 55:
            return "⚖️ NEUTRAL"
        else:
            return "🚫 AVOID"
    
    @st.cache_data(ttl=1800, show_spinner=False)
    def find_top_gems_cached(_self, min_score=65, top_n=25, sector_filter=None):
        """Cached version of find top gems"""
        return _self._find_top_gems(min_score, top_n, sector_filter)
    
    def _find_top_gems(self, min_score=65, top_n=25, sector_filter=None):
        """Find top hidden gems"""
        if self.df.empty:
            return pd.DataFrame()
        
        # Get unique stocks
        unique_stocks = self.df['Stock Code'].unique()
        stocks_to_analyze = unique_stocks[:Config.MAX_STOCKS_ANALYZED]
        
        print(f"🔍 Analyzing {len(stocks_to_analyze)} stocks...")
        results = []
        
        for i, stock in enumerate(stocks_to_analyze):
            if i % 20 == 0:
                print(f"   Progress: {i+1}/{len(stocks_to_analyze)} stocks")
            
            try:
                score_data = self.calculate_gem_score_cached(stock, 90)
                
                if score_data and score_data['total_score'] >= min_score:
                    # Apply sector filter
                    if sector_filter and sector_filter != 'All':
                        if score_data['sector'] != sector_filter:
                            continue
                    
                    # Check liquidity
                    stock_data = self.df[self.df['Stock Code'] == stock]
                    if not stock_data.empty:
                        avg_value = stock_data['Value'].mean() if 'Value' in stock_data.columns else 0
                        if avg_value < 500e6:  # < 500M
                            continue
                    
                    results.append({
                        'Stock': stock,
                        'Score': score_data['total_score'],
                        'Signal': score_data['signal'],
                        'Sector': score_data['sector'],
                        'Price': score_data['latest_price'],
                        'Price Chg %': score_data['price_change'],
                        'Free Float %': score_data['free_float'],
                        'SM Flow (B)': score_data['smart_money_total'] / 1e9,
                        'Positive Months': score_data['positive_months'],
                        'Total Months': score_data['total_months'],
                        'RSI': score_data['rsi'],
                        'Market Trend': score_data['market_trend']
                    })
                    
            except Exception as e:
                self.error_handler.logger.error(f"Error processing {stock}: {e}")
                continue
        
        if results:
            df_results = pd.DataFrame(results)
            return df_results.sort_values('Score', ascending=False).head(top_n)
        
        return pd.DataFrame()

# ==============================================================================
# 📊 VISUALIZATION
# ==============================================================================
class Visualization:
    """Visualization functions"""
    
    @staticmethod
    def create_radar_chart(components, stock_code, signal):
        """Create radar chart for component scores"""
        categories = list(components.keys())
        values = list(components.values())
        
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
                radialaxis=dict(visible=True, range=[0, 100])
            ),
            showlegend=False,
            title=f"📊 {stock_code} - {signal}",
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_price_chart(df, stock_code):
        """Create price chart with moving averages"""
        stock_data = df[df['Stock Code'] == stock_code].sort_values('Date')
        
        if stock_data.empty:
            return None
        
        fig = go.Figure()
        
        # Price line
        fig.add_trace(go.Scatter(
            x=stock_data['Date'],
            y=stock_data['Close'],
            mode='lines',
            name='Price',
            line=dict(color='#4318FF', width=3)
        ))
        
        # Calculate MA20 if enough data
        if len(stock_data) >= 20:
            stock_data['MA20'] = stock_data['Close'].rolling(window=20).mean()
            fig.add_trace(go.Scatter(
                x=stock_data['Date'],
                y=stock_data['MA20'],
                mode='lines',
                name='MA20',
                line=dict(color='#FFB547', width=2, dash='dash')
            ))
        
        fig.update_layout(
            title=f"📈 {stock_code} - Price History",
            height=400,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def create_smart_money_chart(df, stock_code):
        """Create smart money flow chart"""
        stock_data = df[df['Stock Code'] == stock_code].sort_values('Date')
        
        if stock_data.empty or 'Smart_Money_Flow' not in stock_data.columns:
            return None
        
        # Get month-end data
        month_ends = stock_data[stock_data['Date'].dt.is_month_end]
        if month_ends.empty:
            month_ends = stock_data.copy()
        
        fig = go.Figure()
        
        # Smart money bars
        colors = ['#05CD99' if x > 0 else '#EE5D50' for x in month_ends['Smart_Money_Flow']]
        fig.add_trace(go.Bar(
            x=month_ends['Date'],
            y=month_ends['Smart_Money_Flow'] / 1e9,
            name='Smart Money Flow (B)',
            marker_color=colors,
            opacity=0.8
        ))
        
        fig.update_layout(
            title=f"💰 {stock_code} - Smart Money Flow",
            height=400,
            showlegend=False,
            yaxis_title="Flow (Billion Rp)"
        )
        
        return fig
    
    @staticmethod
    def create_sector_heatmap(gems_df):
        """Create heatmap of gems by sector"""
        if gems_df.empty or 'Sector' not in gems_df.columns:
            return None
        
        sector_stats = gems_df.groupby('Sector').agg({
            'Stock': 'count',
            'Score': 'mean',
            'SM Flow (B)': 'sum'
        }).reset_index()
        
        sector_stats.columns = ['Sector', 'Count', 'Avg Score', 'Total Smart Money (B)']
        
        fig = px.treemap(
            sector_stats,
            path=['Sector'],
            values='Total Smart Money (B)',
            color='Avg Score',
            color_continuous_scale='RdYlGn',
            title='🏭 Hidden Gems by Sector',
            hover_data=['Count', 'Avg Score']
        )
        
        fig.update_layout(height=500)
        return fig

# ==============================================================================
# 🚀 MAIN APPLICATION
# ==============================================================================
def main():
    """Main application function"""
    
    # ============================================
    # 1. TAMPILKAN HEADER
    # ============================================
    st.markdown("""
    <div class="header-gradient">
        <div class="header-title">🚀 HIDDEN GEM FINDER v3.0</div>
        <div class="header-subtitle">Enterprise Edition • Smart Money Analytics</div>
    </div>
    """, unsafe_allow_html=True)
    
    # ============================================
    # 2. INISIALISASI DAN LOAD DATA
    # ============================================
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = OptimizedDataLoader()
    
    loader = st.session_state.data_loader
    
    # Load data jika belum ada di session state
    if 'df_merged' not in st.session_state:
        loading_placeholder = st.empty()
        
        with loading_placeholder.container():
            st.info("🔄 **Loading data from Google Drive...**")
            progress_bar = st.progress(0)
            
            try:
                # Step 1: Initialize
                progress_bar.progress(10)
                
                if not loader.service:
                    st.error("❌ Google Drive initialization failed")
                    st.stop()
                
                # Step 2: Load data
                progress_bar.progress(30)
                df_merged = loader.load_all_data()
                
                progress_bar.progress(70)
                
                if df_merged.empty:
                    st.error("❌ No data loaded. Please check your files.")
                    st.stop()
                
                # Step 3: Store in session state
                st.session_state.df_merged = df_merged
                st.session_state.data_loaded = True
                
                progress_bar.progress(100)
                time.sleep(0.5)
                
                # Clear loading
                loading_placeholder.empty()
                
                # Success message
                latest_date = df_merged['Date'].max().strftime('%d %b %Y') if not df_merged.empty else 'N/A'
                st.toast(f"✅ Data loaded! {len(df_merged):,} rows | Latest: {latest_date}", icon="🎉")
                
            except Exception as e:
                loading_placeholder.error(f"❌ Error: {str(e)}")
                st.stop()
    else:
        df_merged = st.session_state.df_merged
    
    # ============================================
    # 3. SIDEBAR SETTINGS
    # ============================================
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=80)
        st.markdown("## ⚙️ Settings")
        st.divider()
        
        # Analysis settings
        lookback_days = st.slider("Analysis Period (Days)", 30, 180, 90, 15)
        min_score = st.slider("Minimum Gem Score", 50, 90, 70, 5)
        top_n = st.slider("Top N Results", 5, 50, 25, 5)
        
        # Sector filter
        sectors = ['All'] + sorted(df_merged['Sector'].dropna().unique().tolist())
        selected_sector = st.selectbox("Filter by Sector", sectors)
        
        # Advanced filters
        with st.expander("🔍 Advanced Filters"):
            min_sm_flow = st.number_input("Min Smart Money (B)", 0.0, 100.0, 1.0, 0.5)
            max_free_float = st.number_input("Max Free Float %", 0.0, 100.0, 60.0, 5.0)
            min_rsi = st.slider("Min RSI", 0, 100, 30, 5)
            max_rsi = st.slider("Max RSI", 0, 100, 70, 5)
        
        st.divider()
        
        # Actions
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🔄 Refresh", use_container_width=True):
                st.cache_data.clear()
                st.session_state.clear()
                st.rerun()
        
        with col_btn2:
            run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
        
        st.divider()
        
        # Data info
        with st.expander("📊 Data Info"):
            st.write(f"**Total Rows:** {len(df_merged):,}")
            st.write(f"**Unique Stocks:** {df_merged['Stock Code'].nunique()}")
            st.write(f"**Date Range:**")
            st.write(f"  {df_merged['Date'].min().date()} to {df_merged['Date'].max().date()}")
            st.write(f"**Latest Date:** {df_merged['Date'].max().date()}")
            
            # KSEI coverage
            if 'Smart_Money_Flow' in df_merged.columns:
                ksei_coverage = (df_merged['Smart_Money_Flow'].notna().sum() / len(df_merged) * 100)
                st.write(f"**KSEI Coverage:** {ksei_coverage:.1f}%")
    
    # ============================================
    # 4. MAIN TABS
    # ============================================
    tab1, tab2, tab3 = st.tabs(["🏆 Top Gems", "📈 Stock Analyzer", "📊 Market Intel"])
    
    # ============================================
    # TAB 1: TOP GEMS
    # ============================================
    with tab1:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">💎 Hidden Gem Candidates</div>', unsafe_allow_html=True)
        
        # Run analysis button
        if run_analysis:
            st.session_state.run_gem_analysis = True
        
        if 'run_gem_analysis' in st.session_state and st.session_state.run_gem_analysis:
            with st.spinner(f"🔍 Analyzing stocks with score ≥ {min_score}..."):
                # Initialize analyzer
                analyzer = GemAnalyzer(df_merged)
                
                # Run analysis
                sector_filter = None if selected_sector == 'All' else selected_sector
                results_df = analyzer.find_top_gems_cached(
                    min_score=min_score,
                    top_n=top_n,
                    sector_filter=sector_filter
                )
                
                # Apply additional filters
                if not results_df.empty:
                    # Apply advanced filters
                    if min_sm_flow > 0:
                        results_df = results_df[results_df['SM Flow (B)'] >= min_sm_flow]
                    
                    if max_free_float < 100:
                        results_df = results_df[results_df['Free Float %'] <= max_free_float]
                    
                    if min_rsi > 0:
                        results_df = results_df[results_df['RSI'] >= min_rsi]
                    
                    if max_rsi < 100:
                        results_df = results_df[results_df['RSI'] <= max_rsi]
                
                # Store results
                st.session_state.gem_results = results_df
        
        # Display results if available
        if 'gem_results' in st.session_state:
            results_df = st.session_state.gem_results
            
            if not results_df.empty:
                # Summary metrics
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                with col_s1:
                    st.metric("Total Gems", len(results_df))
                with col_s2:
                    st.metric("Avg Score", f"{results_df['Score'].mean():.1f}")
                with col_s3:
                    st.metric("Top Sector", results_df['Sector'].mode().iloc[0])
                with col_s4:
                    total_sm = results_df['SM Flow (B)'].sum()
                    st.metric("Total SM Flow", f"{total_sm:.1f}B")
                
                # Results table
                st.markdown("#### 📋 Gem Candidates")
                
                # Format dataframe
                display_df = results_df.copy()
                
                # Function to color signal
                def color_signal(val):
                    if 'STRONG BUY' in val:
                        return 'background-color: #D1FAE5; color: #065F46; font-weight: bold;'
                    elif 'BUY' in val:
                        return 'background-color: #FEF3C7; color: #92400E; font-weight: bold;'
                    elif 'WATCH' in val:
                        return 'background-color: #DBEAFE; color: #1E40AF; font-weight: bold;'
                    elif 'LOW LIQUIDITY' in val:
                        return 'background-color: #FEE2E2; color: #991B1B; font-weight: bold;'
                    else:
                        return ''
                
                st.dataframe(
                    display_df.style
                    .format({
                        'Price': '{:,.0f}',
                        'Price Chg %': '{:.1f}%',
                        'Free Float %': '{:.1f}',
                        'SM Flow (B)': '{:.2f}',
                        'Score': '{:.1f}',
                        'RSI': '{:.1f}'
                    })
                    .applymap(color_signal, subset=['Signal'])
                    .background_gradient(subset=['Score'], cmap='RdYlGn', vmin=70, vmax=100)
                    .bar(subset=['SM Flow (B)'], color='#05CD99'),
                    use_container_width=True,
                    height=500
                )
                
                # Download button
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"hidden_gems_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                # Sector visualization
                st.markdown("#### 🏭 Sector Distribution")
                viz = Visualization()
                sector_chart = viz.create_sector_heatmap(results_df)
                if sector_chart:
                    st.plotly_chart(sector_chart, use_container_width=True)
                
                # Deep dive untuk top gem
                if len(results_df) > 0:
                    top_stock = results_df.iloc[0]['Stock']
                    
                    st.markdown(f"#### 🎯 Deep Dive: {top_stock}")
                    
                    # Get detailed analysis
                    analyzer = GemAnalyzer(df_merged)
                    score_data = analyzer.calculate_gem_score_cached(top_stock, lookback_days)
                    
                    if score_data:
                        col_d1, col_d2 = st.columns(2)
                        
                        with col_d1:
                            # Radar chart
                            radar = viz.create_radar_chart(
                                score_data['components'],
                                top_stock,
                                score_data['signal']
                            )
                            st.plotly_chart(radar, use_container_width=True)
                        
                        with col_d2:
                            # Price chart
                            price_chart = viz.create_price_chart(df_merged, top_stock)
                            if price_chart:
                                st.plotly_chart(price_chart, use_container_width=True)
                        
                        # Smart money chart
                        sm_chart = viz.create_smart_money_chart(df_merged, top_stock)
                        if sm_chart:
                            st.plotly_chart(sm_chart, use_container_width=True)
                        
                        # Metrics
                        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                        with col_m1:
                            st.metric("Gem Score", f"{score_data['total_score']:.1f}")
                        with col_m2:
                            st.metric("Signal", score_data['signal'])
                        with col_m3:
                            st.metric("Price", f"Rp {score_data['latest_price']:,.0f}")
                        with col_m4:
                            st.metric("Price Change", f"{score_data['price_change']:.1f}%")
            else:
                st.info("📭 No gems found with current filters. Try lowering the minimum score.")
        else:
            st.info("👈 Click 'Run Analysis' button to start")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ============================================
    # TAB 2: STOCK ANALYZER
    # ============================================
    with tab2:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📈 Individual Stock Analysis</div>', unsafe_allow_html=True)
        
        # Stock selector
        stocks = sorted(df_merged['Stock Code'].unique())
        selected_stock = st.selectbox("Select Stock", stocks, key="stock_analyzer_select")
        
        if selected_stock:
            # Get stock data
            stock_data = df_merged[df_merged['Stock Code'] == selected_stock].sort_values('Date')
            
            if not stock_data.empty:
                # Latest info
                latest = stock_data.iloc[-1]
                
                col_i1, col_i2, col_i3 = st.columns(3)
                with col_i1:
                    price = latest.get('Close', 0)
                    st.metric("Current Price", f"Rp {price:,.0f}")
                with col_i2:
                    change = latest.get('Change %', 0)
                    st.metric("Daily Change", f"{change:+.2f}%")
                with col_i3:
                    st.metric("Sector", latest.get('Sector', 'N/A'))
                
                # Analyze button
                if st.button("🔍 Analyze Stock", type="primary", use_container_width=True):
                    with st.spinner("Analyzing..."):
                        analyzer = GemAnalyzer(df_merged)
                        score_data = analyzer.calculate_gem_score_cached(selected_stock, lookback_days)
                        
                        if score_data:
                            # Display results
                            st.markdown("#### 📊 Analysis Results")
                            
                            col_r1, col_r2 = st.columns(2)
                            
                            with col_r1:
                                # Score metrics
                                st.metric("Gem Score", f"{score_data['total_score']:.1f}")
                                st.metric("Signal", score_data['signal'])
                                st.metric("Market Trend", score_data['market_trend'])
                            
                            with col_r2:
                                # Financial metrics
                                st.metric("Free Float", f"{score_data['free_float']:.1f}%")
                                st.metric("RSI", f"{score_data['rsi']:.1f}")
                                st.metric("SM Total", f"Rp {score_data['smart_money_total']/1e9:.2f}B")
                            
                            # Charts
                            viz = Visualization()
                            
                            col_c1, col_c2 = st.columns(2)
                            with col_c1:
                                radar = viz.create_radar_chart(
                                    score_data['components'],
                                    selected_stock,
                                    score_data['signal']
                                )
                                st.plotly_chart(radar, use_container_width=True)
                            
                            with col_c2:
                                price_chart = viz.create_price_chart(df_merged, selected_stock)
                                if price_chart:
                                    st.plotly_chart(price_chart, use_container_width=True)
                            
                            # Smart money chart
                            sm_chart = viz.create_smart_money_chart(df_merged, selected_stock)
                            if sm_chart:
                                st.plotly_chart(sm_chart, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ============================================
    # TAB 3: MARKET INTELLIGENCE
    # ============================================
    with tab3:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📊 Market Intelligence</div>', unsafe_allow_html=True)
        
        # Latest market data
        latest_date = df_merged['Date'].max()
        daily_data = df_merged[df_merged['Date'] == latest_date]
        
        if not daily_data.empty:
            # Market metrics
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                advancers = (daily_data['Change %'] > 0).sum()
                decliners = (daily_data['Change %'] < 0).sum()
                st.metric("Advancers/Decliners", f"{advancers}/{decliners}")
            
            with col_m2:
                adv_dec_ratio = advancers / max(decliners, 1)
                st.metric("A/D Ratio", f"{adv_dec_ratio:.2f}")
            
            with col_m3:
                avg_change = daily_data['Change %'].mean()
                st.metric("Avg Change", f"{avg_change:.2f}%")
            
            with col_m4:
                total_value = daily_data['Value'].sum() / 1e12
                st.metric("Total Value", f"Rp {total_value:.2f}T")
            
            # Top gainers/losers
            col_g1, col_g2 = st.columns(2)
            
            with col_g1:
                st.markdown("##### 📈 Top 10 Gainers")
                top_gainers = daily_data.nlargest(10, 'Change %')[['Stock Code', 'Close', 'Change %', 'Value']].copy()
                top_gainers['Value (B)'] = top_gainers['Value'] / 1e9
                
                st.dataframe(
                    top_gainers[['Stock Code', 'Close', 'Change %', 'Value (B)']]
                    .style.format({
                        'Close': '{:,.0f}',
                        'Change %': '{:+.2f}%',
                        'Value (B)': '{:.2f}'
                    })
                    .background_gradient(subset=['Change %'], cmap='Greens'),
                    use_container_width=True,
                    height=350
                )
            
            with col_g2:
                st.markdown("##### 📉 Top 10 Losers")
                top_losers = daily_data.nsmallest(10, 'Change %')[['Stock Code', 'Close', 'Change %', 'Value']].copy()
                top_losers['Value (B)'] = top_losers['Value'] / 1e9
                
                st.dataframe(
                    top_losers[['Stock Code', 'Close', 'Change %', 'Value (B)']]
                    .style.format({
                        'Close': '{:,.0f}',
                        'Change %': '{:+.2f}%',
                        'Value (B)': '{:.2f}'
                    })
                    .background_gradient(subset=['Change %'], cmap='Reds'),
                    use_container_width=True,
                    height=350
                )
            
            # Sector performance
            st.markdown("##### 🏭 Sector Performance")
            sector_perf = daily_data.groupby('Sector').agg({
                'Change %': 'mean',
                'Stock Code': 'count',
                'Value': 'sum'
            }).reset_index()
            
            sector_perf.columns = ['Sector', 'Avg Change %', 'Stock Count', 'Total Value']
            sector_perf['Total Value (T)'] = sector_perf['Total Value'] / 1e12
            sector_perf = sector_perf.sort_values('Avg Change %', ascending=False)
            
            fig_sector = px.bar(
                sector_perf,
                x='Sector',
                y='Avg Change %',
                color='Avg Change %',
                color_continuous_scale='RdYlGn',
                title='Average Daily Change by Sector',
                hover_data=['Stock Count', 'Total Value (T)']
            )
            
            st.plotly_chart(fig_sector, use_container_width=True)
            
            # Volume leaders
            st.markdown("##### 💰 Top 10 by Trading Value")
            value_leaders = daily_data.nlargest(10, 'Value')[['Stock Code', 'Close', 'Change %', 'Value']].copy()
            value_leaders['Value (B)'] = value_leaders['Value'] / 1e9
            
            st.dataframe(
                value_leaders[['Stock Code', 'Close', 'Change %', 'Value (B)']]
                .style.format({
                    'Close': '{:,.0f}',
                    'Change %': '{:+.2f}%',
                    'Value (B)': '{:.2f}'
                })
                .bar(subset=['Value (B)'], color='#4318FF'),
                use_container_width=True,
                height=350
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ============================================
    # FOOTER
    # ============================================
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #A3AED0; font-size: 14px; padding: 1rem;'>"
        f"💎 HIDDEN GEM FINDER v3.0 • Data as of {df_merged['Date'].max().strftime('%d %b %Y')} • "
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
        st.info("Please refresh the page or check your credentials.")
        import traceback
        with st.expander("Technical Details"):
            st.code(traceback.format_exc())
