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
    /* Progress bar styling */
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
# 📦 DATA LOADER CLASS
# ==============================================================================
class DataLoader:
    def __init__(self):
        self.service = None
        self.initialize_gdrive()
    
    def initialize_gdrive(self):
        """Initialize Google Drive service with Streamlit secrets"""
        try:
            creds_json = st.secrets["gcp_service_account"]
            creds = Credentials.from_service_account_info(
                creds_json, 
                scopes=['https://www.googleapis.com/auth/drive.readonly']
            )
            self.service = build('drive', 'v3', credentials=creds, cache_discovery=False)
            return True
        except Exception as e:
            st.error(f"❌ GDrive Auth Error: {e}")
            return False
    
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
            st.error(error)
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
            st.error(error)
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(fh, dtype=object)
            
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
            
            return df
            
        except Exception as e:
            st.error(f"Error processing historical data: {e}")
            return pd.DataFrame()
    
    @st.cache_data(ttl=3600, show_spinner="🔄 Merging Datasets...")
    def load_merged_data(_self):
        """Intelligent merge of both datasets"""
        with st.spinner("Loading datasets in parallel..."):
            # Load both datasets in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_ksei = executor.submit(_self.load_ksei_data)
                future_hist = executor.submit(_self.load_historical_data)
                
                df_ksei = future_ksei.result()
                df_hist = future_hist.result()
        
        if df_ksei.empty or df_hist.empty:
            st.error("One or both datasets failed to load")
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
            
            # Forward fill KSEI data (monthly) for daily continuity
            ownership_cols = [c for c in OWNERSHIP_COLS if c in merged.columns]
            ksei_cols = ['Total_chg_Rp', 'Smart_Money_Flow', 'Retail_Flow', 'Institutional_Net'] + ownership_cols
            
            merged = merged.sort_values(['Stock Code', 'Date'])
            for col in ksei_cols:
                if col in merged.columns:
                    merged[col] = merged.groupby('Stock Code')[col].ffill()
            
            # Calculate derived metrics
            merged['Price_Change_1D'] = merged.groupby('Stock Code')['Close'].pct_change()
            merged['Volume_Change_1D'] = merged.groupby('Stock Code')['Volume'].pct_change()
            
            # Money Flow Strength
            if 'Money Flow Value' in merged.columns:
                merged['MF_Strength'] = merged['Money Flow Value'] / merged['Value'].replace(0, 1)
            
            # Ownership Concentration Score
            if 'Free Float' in merged.columns:
                merged['Ownership_Score'] = (100 - merged['Free Float']) / 100  # Higher = more concentrated
            
            st.success(f"✅ Merged dataset: {len(merged):,} rows, {len(merged.columns)} columns")
            return merged
            
        except Exception as e:
            st.error(f"Merge error: {e}")
            return pd.DataFrame()

# ==============================================================================
# 🎯 HIDDEN GEM ANALYZER
# ==============================================================================
class HiddenGemAnalyzer:
    def __init__(self, df):
        self.df = df.copy()
        self.today = self.df['Date'].max()
    
    def calculate_gem_score(self, stock_code, lookback_days=30):
        """Calculate Hidden Gem Score (0-100) for a stock"""
        try:
            # Get data for this stock
            stock_data = self.df[self.df['Stock Code'] == stock_code].sort_values('Date')
            if stock_data.empty:
                return 0
            
            # Get recent data
            cutoff_date = self.today - timedelta(days=lookback_days)
            recent_data = stock_data[stock_data['Date'] >= cutoff_date]
            
            if recent_data.empty:
                return 0
            
            latest = recent_data.iloc[-1]
            
            scores = {}
            
            # 1. SMART MONEY SCORE (35%)
            if 'Smart_Money_Flow' in recent_data.columns and 'Retail_Flow' in recent_data.columns:
                smart_total = recent_data['Smart_Money_Flow'].sum()
                retail_total = recent_data['Retail_Flow'].sum()
                
                if abs(smart_total) > 0:
                    # Score based on smart money vs retail divergence
                    smart_score = min(100, abs(smart_total) / 1e9 * 2)  # Scale by 1B
                    
                    # Bonus for divergence (smart buying, retail selling)
                    if smart_total > 0 and retail_total < 0:
                        smart_score += 20
                    elif smart_total > retail_total * 2:  # Smart dominating
                        smart_score += 15
                    
                    scores['smart_money'] = min(100, smart_score)
                else:
                    scores['smart_money'] = 30
            else:
                scores['smart_money'] = 30
            
            # 2. TECHNICAL SCORE (30%)
            tech_score = 50  # Base
            
            # Price momentum
            if len(recent_data) > 5:
                price_change = (latest['Close'] - recent_data.iloc[0]['Close']) / recent_data.iloc[0]['Close'] * 100
                if -5 <= price_change <= 10:  # Sweet spot: not too high, not crashing
                    tech_score += 20
                elif price_change < -5:
                    tech_score += max(0, 30 + price_change)  # Oversold bonus
            
            # Volume confirmation
            if 'Volume' in recent_data.columns:
                avg_volume = recent_data['Volume'].mean()
                if latest['Volume'] > avg_volume * 1.5:
                    tech_score += 10
            
            # Money Flow
            if 'Money Flow Ratio (20D)' in latest:
                if latest['Money Flow Ratio (20D)'] > 1.2:
                    tech_score += 15
                elif latest['Money Flow Ratio (20D)'] < 0.8:
                    tech_score -= 10
            
            scores['technical'] = min(100, max(0, tech_score))
            
            # 3. FUNDAMENTAL SCORE (25%)
            funda_score = 50
            
            # Free Float analysis (20-40% is optimal)
            if 'Free Float' in latest:
                ff = latest['Free Float']
                if 20 <= ff <= 40:
                    funda_score += 25
                elif ff < 20:  # Very concentrated
                    funda_score += 15
                elif ff > 60:  # Too diluted
                    funda_score -= 10
            
            # Market Cap proxy
            if 'Listed Shares' in latest and latest['Close'] > 0:
                mcap = latest['Close'] * latest['Listed Shares']
                if mcap > 1e12:  # > 1T IDR
                    funda_score += 10  # Large cap stability
            
            scores['fundamental'] = min(100, max(0, funda_score))
            
            # 4. SENTIMENT SCORE (10%)
            sentiment_score = 50
            
            # Institutional net flow
            if 'Institutional_Net' in recent_data.columns:
                inst_net = recent_data['Institutional_Net'].sum()
                if inst_net > 1e9:  # 1B IDR net inflow
                    sentiment_score += 30
                elif inst_net < -1e9:
                    sentiment_score -= 20
            
            # Unusual volume
            if 'Unusual Volume' in latest and latest['Unusual Volume']:
                sentiment_score += 10
            
            scores['sentiment'] = min(100, max(0, sentiment_score))
            
            # Calculate weighted total
            total_score = (
                scores['smart_money'] * SCORE_WEIGHTS['smart_money'] +
                scores['technical'] * SCORE_WEIGHTS['technical'] +
                scores['fundamental'] * SCORE_WEIGHTS['fundamental'] +
                scores['sentiment'] * SCORE_WEIGHTS['sentiment']
            )
            
            return {
                'total_score': round(total_score, 1),
                'component_scores': scores,
                'latest_price': latest['Close'],
                'sector': latest.get('Sector', 'N/A'),
                'free_float': latest.get('Free Float', 0),
                'smart_money_30d': recent_data['Smart_Money_Flow'].sum() if 'Smart_Money_Flow' in recent_data.columns else 0,
                'retail_30d': recent_data['Retail_Flow'].sum() if 'Retail_Flow' in recent_data.columns else 0
            }
            
        except Exception as e:
            st.error(f"Error calculating score for {stock_code}: {e}")
            return {'total_score': 0, 'component_scores': {}}
    
    def find_top_gems(self, top_n=20, min_score=60):
        """Find top hidden gem candidates"""
        unique_stocks = self.df['Stock Code'].unique()
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, stock in enumerate(unique_stocks[:100]):  # Limit to 100 for performance
            status_text.text(f"🔍 Analyzing {stock}... ({i+1}/{min(100, len(unique_stocks))})")
            
            score_data = self.calculate_gem_score(stock)
            if score_data['total_score'] >= min_score:
                results.append({
                    'Stock Code': stock,
                    'Gem Score': score_data['total_score'],
                    'Sector': score_data['sector'],
                    'Price': score_data['latest_price'],
                    'Free Float %': score_data['free_float'],
                    'Smart Money 30D (B)': score_data['smart_money_30d'] / 1e9,
                    'Retail 30D (B)': score_data['retail_30d'] / 1e9,
                    'Smart Score': score_data['component_scores'].get('smart_money', 0),
                    'Tech Score': score_data['component_scores'].get('technical', 0),
                    'Fundamental Score': score_data['component_scores'].get('fundamental', 0)
                })
            
            progress_bar.progress((i + 1) / min(100, len(unique_stocks)))
        
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
def create_gem_radar_chart(scores, stock_code):
    """Create radar chart for gem score components"""
    categories = ['Smart Money', 'Technical', 'Fundamental', 'Sentiment']
    values = [
        scores.get('smart_money', 0),
        scores.get('technical', 0),
        scores.get('fundamental', 0),
        scores.get('sentiment', 0)
    ]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values + [values[0]],  # Close the loop
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(67, 24, 255, 0.3)',
        line=dict(color='#4318FF', width=2)
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
        title=f"Score Components: {stock_code}",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#2B3674')
    )
    
    return fig

def create_ownership_flow_chart(df, stock_code):
    """Create ownership flow chart for a stock"""
    stock_data = df[df['Stock Code'] == stock_code].sort_values('Date')
    
    if stock_data.empty:
        return None
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Price vs Smart Money Flow', 'Institutional vs Retail Flow'),
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4]
    )
    
    # Price and Smart Money
    fig.add_trace(
        go.Scatter(x=stock_data['Date'], y=stock_data['Close'], 
                  name='Price', line=dict(color='#4318FF')),
        row=1, col=1
    )
    
    if 'Smart_Money_Flow' in stock_data.columns:
        fig.add_trace(
            go.Bar(x=stock_data['Date'], y=stock_data['Smart_Money_Flow'] / 1e9,
                  name='Smart Money (B)', marker_color='#05CD99', opacity=0.6),
            row=1, col=1
        )
    
    # Institutional vs Retail
    if 'Institutional_Net' in stock_data.columns and 'Retail_Flow' in stock_data.columns:
        fig.add_trace(
            go.Bar(x=stock_data['Date'], y=stock_data['Institutional_Net'] / 1e9,
                  name='Institutional Net (B)', marker_color='#4318FF'),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(x=stock_data['Date'], y=stock_data['Retail_Flow'] / 1e9,
                  name='Retail Flow (B)', marker_color='#EE5D50'),
            row=2, col=1
        )
    
    fig.update_layout(
        height=600,
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#2B3674'),
        hovermode='x unified'
    )
    
    fig.update_xaxes(showgrid=True, gridcolor='#E0E5F2', tickfont=dict(color='#A3AED0'))
    fig.update_yaxes(showgrid=True, gridcolor='#E0E5F2', tickfont=dict(color='#A3AED0'))
    
    return fig

# ==============================================================================
# 🎨 MAIN DASHBOARD
# ==============================================================================
def main():
    # Header
    st.markdown("""
    <div class="header-banner">
        <div class="header-title">🚀 MERGE ANALYTIC KSEI & DAILY TX1Y</div>
        <div class="header-subtitle">Hidden Gem Finder • Institutional Flow Analysis • Multi-dimensional Scoring</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize data loader
    loader = DataLoader()
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2910/2910312.png", width=60)
        st.markdown("<h3 style='color:#2B3674;'>🧠 Smart Analytics</h3>", unsafe_allow_html=True)
        st.divider()
        
        st.markdown("##### 📊 Data Controls")
        if st.button("🔄 Refresh All Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.divider()
        
        st.markdown("##### 🎯 Gem Finder Settings")
        lookback_days = st.slider("Lookback Period (Days)", 7, 90, 30)
        min_gem_score = st.slider("Minimum Gem Score", 0, 100, 60)
        top_n_gems = st.slider("Top N Gems", 5, 50, 20)
        
        st.divider()
        
        st.markdown("##### ⚙️ Advanced")
        show_raw_data = st.checkbox("Show Raw Data", False)
    
    # Load data with progress
    with st.spinner("🚀 Loading and merging datasets..."):
        df_merged = loader.load_merged_data()
    
    if df_merged.empty:
        st.error("Failed to load data. Please check credentials and file access.")
        return
    
    # Dashboard Stats
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_stocks = df_merged['Stock Code'].nunique()
        st.metric("Total Stocks", f"{total_stocks:,}")
    
    with col2:
        latest_date = df_merged['Date'].max().strftime('%d %b %Y')
        st.metric("Latest Data", latest_date)
    
    with col3:
        total_days = (df_merged['Date'].max() - df_merged['Date'].min()).days
        st.metric("Data Period", f"{total_days} days")
    
    with col4:
        active_sectors = df_merged['Sector'].nunique()
        st.metric("Sectors", active_sectors)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏆 Hidden Gems", "📈 Stock Analyzer", "📊 Market Overview", 
        "🔄 Sector Rotation", "📁 Data Explorer"
    ])
    
    # Tab 1: Hidden Gems
    with tab1:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">💎 Top Hidden Gem Candidates</div>', unsafe_allow_html=True)
        
        analyzer = HiddenGemAnalyzer(df_merged)
        
        if st.button("🔍 Find Hidden Gems", type="primary"):
            with st.spinner(f"Analyzing stocks with min score {min_gem_score}..."):
                gems_df = analyzer.find_top_gems(top_n=top_n_gems, min_score=min_gem_score)
            
            if not gems_df.empty:
                # Display gems
                st.dataframe(
                    gems_df.style.format({
                        'Price': '{:,.0f}',
                        'Free Float %': '{:.1f}%',
                        'Smart Money 30D (B)': '{:.2f}',
                        'Retail 30D (B)': '{:.2f}',
                        'Gem Score': '{:.1f}',
                        'Smart Score': '{:.1f}',
                        'Tech Score': '{:.1f}',
                        'Fundamental Score': '{:.1f}'
                    }).background_gradient(
                        subset=['Gem Score'], 
                        cmap='RdYlGn', 
                        vmin=60, 
                        vmax=100
                    ),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Visualize top gem
                if len(gems_df) > 0:
                    top_gem = gems_df.iloc[0]['Stock Code']
                    score_data = analyzer.calculate_gem_score(top_gem)
                    
                    col_chart1, col_chart2 = st.columns(2)
                    
                    with col_chart1:
                        st.plotly_chart(
                            create_gem_radar_chart(score_data['component_scores'], top_gem),
                            use_container_width=True
                        )
                    
                    with col_chart2:
                        st.plotly_chart(
                            create_ownership_flow_chart(df_merged, top_gem),
                            use_container_width=True
                        )
            else:
                st.info("No hidden gems found with the current criteria. Try lowering the minimum score.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 2: Stock Analyzer
    with tab2:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🔍 Individual Stock Analysis</div>', unsafe_allow_html=True)
        
        # Stock selector
        available_stocks = sorted(df_merged['Stock Code'].unique())
        selected_stock = st.selectbox("Select Stock", available_stocks, 
                                     index=available_stocks.index('BBRI') if 'BBRI' in available_stocks else 0)
        
        if selected_stock:
            # Calculate score
            score_data = analyzer.calculate_gem_score(selected_stock)
            
            # Display metrics
            col_score1, col_score2, col_score3, col_score4 = st.columns(4)
            
            with col_score1:
                st.metric("Gem Score", f"{score_data['total_score']:.1f}/100")
            
            with col_score2:
                st.metric("Smart Money 30D", 
                         f"Rp {score_data['smart_money_30d']/1e9:.1f}B",
                         delta=f"Rp {score_data['retail_30d']/1e9:.1f}B Retail")
            
            with col_score3:
                st.metric("Latest Price", f"Rp {score_data['latest_price']:,.0f}")
            
            with col_score4:
                st.metric("Free Float", f"{score_data['free_float']:.1f}%")
            
            # Charts
            st.plotly_chart(
                create_ownership_flow_chart(df_merged, selected_stock),
                use_container_width=True
            )
            
            # Detailed scores
            with st.expander("📊 Detailed Score Breakdown"):
                scores_df = pd.DataFrame.from_dict(
                    score_data['component_scores'], 
                    orient='index', 
                    columns=['Score']
                )
                scores_df['Weight'] = [SCORE_WEIGHTS.get(k.replace(' ', '_').lower(), 0) 
                                      for k in scores_df.index]
                scores_df['Weighted Score'] = scores_df['Score'] * scores_df['Weight']
                
                st.dataframe(
                    scores_df.style.format({
                        'Score': '{:.1f}',
                        'Weight': '{:.2f}',
                        'Weighted Score': '{:.2f}'
                    }),
                    use_container_width=True
                )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 3: Market Overview
    with tab3:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📈 Market Overview</div>', unsafe_allow_html=True)
        
        # Date selector for market overview
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
                col_ov1, col_ov2 = st.columns(2)
                
                with col_ov1:
                    # Top gainers
                    top_gainers = daily_data.nlargest(10, 'Change %')[['Stock Code', 'Close', 'Change %']]
                    st.markdown("##### 📈 Top Gainers")
                    st.dataframe(
                        top_gainers.style.format({
                            'Close': 'Rp {:,.0f}',
                            'Change %': '{:.2f}%'
                        }).background_gradient(
                            subset=['Change %'], 
                            cmap='Greens'
                        ),
                        use_container_width=True,
                        hide_index=True
                    )
                
                with col_ov2:
                    # Top value
                    top_value = daily_data.nlargest(10, 'Value')[['Stock Code', 'Value', 'Volume']]
                    st.markdown("##### 💰 Top Value")
                    st.dataframe(
                        top_value.style.format({
                            'Value': 'Rp {:,.0f}',
                            'Volume': '{:,.0f}'
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
                    
                    fig_sector = px.bar(
                        sector_perf.sort_values('Avg Change %', ascending=False),
                        x='Sector',
                        y='Avg Change %',
                        color='Avg Change %',
                        color_continuous_scale='RdYlGn',
                        title=f"Sector Performance - {selected_date.strftime('%d %b %Y')}"
                    )
                    st.plotly_chart(fig_sector, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 4: Sector Rotation
    with tab4:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🔄 Sector Rotation Analysis</div>', unsafe_allow_html=True)
        
        # Date range selector
        col_dr1, col_dr2 = st.columns(2)
        with col_dr1:
            start_date = st.date_input(
                "Start Date",
                value=df_merged['Date'].max() - timedelta(days=30)
            )
        with col_dr2:
            end_date = st.date_input(
                "End Date",
                value=df_merged['Date'].max()
            )
        
        if start_date and end_date:
            period_data = df_merged[
                (df_merged['Date'] >= pd.Timestamp(start_date)) & 
                (df_merged['Date'] <= pd.Timestamp(end_date))
            ]
            
            if not period_data.empty and 'Smart_Money_Flow' in period_data.columns:
                # Sector-wise smart money flow
                sector_flow = period_data.groupby('Sector').agg({
                    'Smart_Money_Flow': 'sum',
                    'Retail_Flow': 'sum',
                    'Stock Code': 'nunique'
                }).reset_index()
                
                sector_flow.columns = ['Sector', 'Smart Money Flow', 'Retail Flow', 'Stock Count']
                sector_flow['Net Institutional'] = sector_flow['Smart Money Flow'] - sector_flow['Retail Flow']
                
                # Display
                col_sr1, col_sr2 = st.columns(2)
                
                with col_sr1:
                    st.markdown("##### 🏆 Top Sector Inflows")
                    top_inflows = sector_flow.nlargest(10, 'Net Institutional')
                    fig_inflows = px.bar(
                        top_inflows,
                        x='Sector',
                        y='Net Institutional',
                        color='Net Institutional',
                        color_continuous_scale='Greens',
                        title="Top Sector Inflows (Net Institutional)"
                    )
                    st.plotly_chart(fig_inflows, use_container_width=True)
                
                with col_sr2:
                    st.markdown("##### 🔻 Top Sector Outflows")
                    top_outflows = sector_flow.nsmallest(10, 'Net Institutional')
                    fig_outflows = px.bar(
                        top_outflows,
                        x='Sector',
                        y='Net Institutional',
                        color='Net Institutional',
                        color_continuous_scale='Reds',
                        title="Top Sector Outflows (Net Institutional)"
                    )
                    st.plotly_chart(fig_outflows, use_container_width=True)
                
                # Sector heatmap
                st.markdown("##### 🔥 Sector Flow Heatmap")
                heatmap_data = sector_flow[['Sector', 'Smart Money Flow', 'Retail Flow', 'Net Institutional']].copy()
                heatmap_data['Smart Money Flow'] = heatmap_data['Smart Money Flow'] / 1e9
                heatmap_data['Retail Flow'] = heatmap_data['Retail Flow'] / 1e9
                heatmap_data['Net Institutional'] = heatmap_data['Net Institutional'] / 1e9
                
                st.dataframe(
                    heatmap_data.style.format({
                        'Smart Money Flow': '{:.1f}B',
                        'Retail Flow': '{:.1f}B',
                        'Net Institutional': '{:.1f}B'
                    }).background_gradient(
                        subset=['Net Institutional'], 
                        cmap='RdYlGn'
                    ),
                    use_container_width=True,
                    hide_index=True
                )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 5: Data Explorer
    with tab5:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📁 Data Explorer</div>', unsafe_allow_html=True)
        
        if show_raw_data:
            with st.expander("🔍 View Raw Data Sample"):
                st.dataframe(df_merged.head(100), use_container_width=True)
        
        # Data statistics
        st.markdown("##### 📊 Dataset Statistics")
        
        col_stats1, col_stats2, col_stats3 = st.columns(3)
        
        with col_stats1:
            st.metric("Total Rows", f"{len(df_merged):,}")
        
        with col_stats2:
            st.metric("Total Columns", f"{len(df_merged.columns):,}")
        
        with col_stats3:
            missing_pct = (df_merged.isnull().sum().sum() / (len(df_merged) * len(df_merged.columns))) * 100
            st.metric("Missing Data", f"{missing_pct:.1f}%")
        
        # Column explorer
        st.markdown("##### 🗂️ Column Information")
        col_info = pd.DataFrame({
            'Column Name': df_merged.columns,
            'Data Type': df_merged.dtypes.astype(str),
            'Non-Null Count': df_merged.count().values,
            'Unique Values': df_merged.nunique().values
        })
        
        st.dataframe(col_info, use_container_width=True, hide_index=True)
        
        # Sample queries
        st.markdown("##### 🔎 Sample Queries")
        
        query_col1, query_col2 = st.columns(2)
        
        with query_col1:
            if st.button("🔍 Stocks with Highest Smart Money Inflow"):
                if 'Smart_Money_Flow' in df_merged.columns:
                    latest_date = df_merged['Date'].max()
                    latest_data = df_merged[df_merged['Date'] == latest_date]
                    
                    if not latest_data.empty:
                        top_smart = latest_data.nlargest(10, 'Smart_Money_Flow')[
                            ['Stock Code', 'Smart_Money_Flow', 'Close', 'Sector']
                        ]
                        top_smart['Smart_Money_Flow_B'] = top_smart['Smart_Money_Flow'] / 1e9
                        
                        st.dataframe(
                            top_smart[['Stock Code', 'Smart_Money_Flow_B', 'Close', 'Sector']]
                            .style.format({
                                'Close': 'Rp {:,.0f}',
                                'Smart_Money_Flow_B': '{:.1f}B'
                            }),
                            use_container_width=True,
                            hide_index=True
                        )
        
        with query_col2:
            if st.button("📉 Oversold with Smart Money Buying"):
                # Find stocks where price down but smart money buying
                if all(col in df_merged.columns for col in ['Change %', 'Smart_Money_Flow']):
                    latest_date = df_merged['Date'].max()
                    latest_data = df_merged[df_merged['Date'] == latest_date]
                    
                    if not latest_data.empty:
                        oversold_smart = latest_data[
                            (latest_data['Change %'] < -2) & 
                            (latest_data['Smart_Money_Flow'] > 0)
                        ].nlargest(10, 'Smart_Money_Flow')[
                            ['Stock Code', 'Change %', 'Smart_Money_Flow', 'Close']
                        ]
                        
                        oversold_smart['Smart_Money_Flow_B'] = oversold_smart['Smart_Money_Flow'] / 1e9
                        
                        st.dataframe(
                            oversold_smart[['Stock Code', 'Change %', 'Smart_Money_Flow_B', 'Close']]
                            .style.format({
                                'Change %': '{:.2f}%',
                                'Close': 'Rp {:,.0f}',
                                'Smart_Money_Flow_B': '{:.1f}B'
                            }).background_gradient(
                                subset=['Change %'], 
                                cmap='Reds_r'
                            ),
                            use_container_width=True,
                            hide_index=True
                        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #A3AED0; font-size: 14px;'>"
        "🚀 MERGE ANALYTIC KSEI & DAILY TX1Y • Built with Streamlit • "
        f"Data as of {df_merged['Date'].max().strftime('%d %b %Y')}"
        "</div>",
        unsafe_allow_html=True
    )

# ==============================================================================
# 🚀 RUN APP
# ==============================================================================
if __name__ == "__main__":
    main()
