# ==============================================================================
# 🚀 HIDDEN GEM FINDER v4.0 - LITE & FAST EDITION
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

# Google Drive imports
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ==============================================================================
# ⚙️ KONFIGURASI
# ==============================================================================
st.set_page_config(
    page_title="Terminal Saham Pro v4 (Fast)",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

# --- CSS DARK MODE (OPTIMIZED) ---
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; font-family: 'Roboto Mono', monospace; }
    [data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid #30363D; }
    div[data-testid="stMetricValue"] { font-size: 24px !important; color: #00E676 !important; }
    .css-card { background-color: #1E2329; padding: 15px; border-radius: 8px; border: 1px solid #30363D; margin-bottom: 15px; }
    div[data-testid="stDataFrame"] { background-color: #1E2329 !important; }
    h1, h2, h3 { color: #58A6FF !important; font-family: 'Inter', sans-serif; }
    div.stButton > button { background-color: #238636; color: white; border: 1px solid #30363D; }
</style>
""", unsafe_allow_html=True)

class Config:
    FOLDER_ID = "1hX2jwUrAgi4Fr8xkcFWjCW6vbk6lsIlP"
    FILE_KSEI = "KSEI_Shareholder_Processed.csv"
    FILE_HIST = "Kompilasi_Data_1Tahun.csv"
    
    OWNERSHIP_COLS = [
        'Local IS', 'Local CP', 'Local PF', 'Local IB', 'Local ID', 'Local MF', 
        'Foreign IS', 'Foreign CP', 'Foreign PF', 'Foreign IB'
    ]

# ==============================================================================
# 📦 DATA LOADER (OPTIMIZED & ROBUST)
# ==============================================================================
def get_gdrive_service():
    try:
        if "gcp_service_account" not in st.secrets: return None, "Secrets missing"
        creds_data = st.secrets["gcp_service_account"]
        creds_json = creds_data.to_dict() if hasattr(creds_data, "to_dict") else dict(creds_data)
        if "private_key" in creds_json:
            pk = str(creds_json["private_key"])
            if "\\n" in pk: creds_json["private_key"] = pk.replace("\\n", "\n")
        creds = Credentials.from_service_account_info(creds_json, scopes=['https://www.googleapis.com/auth/drive.readonly'])
        return build('drive', 'v3', credentials=creds, cache_discovery=False), None
    except Exception as e: return None, str(e)

@st.cache_data(ttl=3600, show_spinner="🚀 Downloading & Vectorizing Data...")
def load_and_process_data():
    """Fungsi ini memuat data DAN melakukan kalkulasi berat SEKALIGUS (Vectorized)"""
    service, err = get_gdrive_service()
    if err: return pd.DataFrame(), err, "error"

    try:
        # 1. Load Data
        def download_df(fname):
            res = service.files().list(q=f"'{Config.FOLDER_ID}' in parents and name='{fname}'", fields="files(id)").execute()
            if not res.get('files'): return None
            req = service.files().get_media(fileId=res['files'][0]['id'])
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, req)
            done = False
            while not done: _, done = downloader.next_chunk()
            fh.seek(0)
            return pd.read_csv(fh, dtype=str)

        df_ksei = download_df(Config.FILE_KSEI)
        df_hist = download_df(Config.FILE_HIST)

        if df_ksei is None or df_hist is None: return pd.DataFrame(), "File not found", "error"

        # 2. Clean & Process Historical (Daily)
        df_hist.columns = df_hist.columns.str.strip()
        df_hist['Date'] = pd.to_datetime(df_hist['Last Trading Date'], errors='coerce')
        
        num_cols = ['Close', 'Volume', 'Value', 'Net Foreign Flow', 'Money Flow Value']
        for c in num_cols:
            if c in df_hist.columns:
                df_hist[c] = pd.to_numeric(df_hist[c].str.replace(r'[,\sRp\%]', '', regex=True), errors='coerce').fillna(0)
        
        df_hist = df_hist.sort_values(['Stock Code', 'Date'])
        
        # --- VECTORIZED CALCULATIONS (SUPER FAST) ---
        # Hitung indikator untuk SEMUA saham sekaligus
        g = df_hist.groupby('Stock Code')
        
        # RSI 14
        def calc_rsi_vec(x):
            delta = x.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            return 100 - (100 / (1 + rs))
            
        df_hist['RSI'] = g['Close'].transform(calc_rsi_vec).fillna(50)
        
        # Moving Averages
        df_hist['MA20'] = g['Close'].transform(lambda x: x.rolling(20).mean())
        df_hist['MA50'] = g['Close'].transform(lambda x: x.rolling(50).mean())
        
        # Volatility (Annualized)
        df_hist['Volatility'] = g['Close'].transform(lambda x: x.pct_change().rolling(20).std() * np.sqrt(252)).fillna(0)
        
        # Volume Trend
        df_hist['Vol_MA20'] = g['Volume'].transform(lambda x: x.rolling(20).mean())
        df_hist['Vol_Ratio'] = df_hist['Volume'] / df_hist['Vol_MA20'].replace(0, 1)

        # 3. Clean & Process KSEI (Monthly)
        df_ksei['Date'] = pd.to_datetime(df_ksei['Date'], errors='coerce')
        ksei_cols = [c for c in df_ksei.columns if 'chg_Rp' in c]
        for c in ksei_cols:
            df_ksei[c] = pd.to_numeric(df_ksei[c].str.replace(',', ''), errors='coerce').fillna(0)
        
        # Smart Money Calculation (KSEI) - Grouped by Month/Stock
        sm_cols = [c for c in ksei_cols if any(x in c for x in ['Foreign IS', 'Foreign IB', 'Local IS', 'Local PF', 'Local CP'])]
        retail_col = 'Local ID_chg_Rp' if 'Local ID_chg_Rp' in df_ksei.columns else None
        
        df_ksei['Smart_Money_Flow'] = df_ksei[sm_cols].sum(axis=1)
        if retail_col: df_ksei['Retail_Flow'] = df_ksei[retail_col]
        else: df_ksei['Retail_Flow'] = 0
            
        # 4. Merge (Simplified: Left Join on Date)
        # Kita buat kolom 'Month' untuk join
        df_hist['Month'] = df_hist['Date'].dt.to_period('M')
        df_ksei['Month'] = df_ksei['Date'].dt.to_period('M')
        
        # Ambil data KSEI terakhir per bulan per saham
        ksei_agg = df_ksei.groupby(['Stock Code', 'Month'])[['Smart_Money_Flow', 'Retail_Flow']].last().reset_index()
        
        # Merge ke data harian
        df_final = pd.merge(df_hist, ksei_agg, on=['Stock Code', 'Month'], how='left')
        
        # Fill NA flow with 0 (karena KSEI data bulanan)
        df_final['Smart_Money_Flow'] = df_final['Smart_Money_Flow'].fillna(0)
        df_final['Retail_Flow'] = df_final['Retail_Flow'].fillna(0)
        
        # Sector Clean
        df_final['Sector'] = df_final['Sector'].astype(str).str.strip().fillna('Others')

        return df_final, "Data Ready", "success"

    except Exception as e:
        return pd.DataFrame(), str(e), "error"

# ==============================================================================
# 🧠 LOGIC ANALYZER (VECTORIZED)
# ==============================================================================
def analyze_gems(df, min_score=60):
    """
    Fungsi ini berjalan CEPAT karena hanya memproses snapshot terakhir
    dari setiap saham, bukan looping histori.
    """
    # Ambil data tanggal terakhir yang tersedia di dataset
    max_date = df['Date'].max()
    
    # Filter hanya baris data terakhir per saham
    latest = df[df['Date'] == max_date].copy()
    
    if latest.empty: return pd.DataFrame()

    # --- SCORING LOGIC (VECTORIZED) ---
    
    # 1. Smart Money Score (0-40)
    # Kita cari akumulasi 3 bulan terakhir untuk skor yang lebih valid
    start_3m = max_date - timedelta(days=90)
    df_3m = df[df['Date'] >= start_3m]
    sm_3m = df_3m.groupby('Stock Code')['Smart_Money_Flow'].sum()
    
    latest = latest.set_index('Stock Code')
    latest['SM_3M'] = sm_3m
    
    # Skor based on nominal akumulasi
    latest['Score_SM'] = np.where(latest['SM_3M'] > 50e9, 40,
                         np.where(latest['SM_3M'] > 10e9, 30,
                         np.where(latest['SM_3M'] > 0, 15, 0)))
    
    # 2. Technical Score (0-30)
    # RSI: Ideal 40-60 (Accumulation zone), Oversold < 30 (Bounce)
    latest['Score_Tech'] = np.where((latest['RSI'] >= 40) & (latest['RSI'] <= 60), 20,
                           np.where(latest['RSI'] < 30, 30, # Oversold bonus
                           np.where(latest['RSI'] > 70, 0, 10)))
    
    # MA Alignment bonus
    latest['Score_Tech'] += np.where(latest['Close'] > latest['MA20'], 10, 0)
    
    # 3. Volatility Score (0-30) - Low Volatility preferred for stability
    # Volatility < 30% is good
    latest['Score_Vol'] = np.where(latest['Volatility'] < 0.3, 30,
                          np.where(latest['Volatility'] < 0.6, 15, 5))
    
    # TOTAL SCORE
    latest['Gem_Score'] = latest['Score_SM'] + latest['Score_Tech'] + latest['Score_Vol']
    
    # --- BANDARMOLOGY DIVERGENCE ---
    # Harga turun/flat (Return < 5%) TAPI Smart Money Masuk (> 5M)
    # Kita butuh return 1 bulan
    start_1m = max_date - timedelta(days=30)
    price_1m_ago = df[df['Date'] == start_1m].set_index('Stock Code')['Close']
    # Fallback jika tanggal pas libur, ambil average
    if price_1m_ago.empty:
        price_1m_ago = df[df['Date'] >= start_1m].groupby('Stock Code')['Close'].first()
        
    latest['Price_1M_Ago'] = price_1m_ago
    latest['Return_1M'] = (latest['Close'] - latest['Price_1M_Ago']) / latest['Price_1M_Ago'] * 100
    
    latest['Divergence'] = (latest['Return_1M'] < 5) & (latest['SM_3M'] > 5e9)
    
    # Tambah Bonus Skor Divergence
    latest.loc[latest['Divergence'], 'Gem_Score'] += 10
    latest['Gem_Score'] = latest['Gem_Score'].clip(upper=100) # Cap at 100

    # Filter Minimum Score & Liquidity (Anti-Zonk)
    # Wajib ada transaksi > 500 Juta hari ini (atau rata-rata)
    filtered = latest[
        (latest['Gem_Score'] >= min_score) & 
        (latest['Value'] > 500e6) # Filter saham mati
    ]
    
    return filtered.sort_values('Gem_Score', ascending=False).reset_index()

# ==============================================================================
# 🚀 MAIN APP
# ==============================================================================
def main():
    st.markdown("<div class='header-title'>TERMINAL SAHAM PRO v4</div>", unsafe_allow_html=True)
    st.markdown("<div class='header-subtitle'>Lite Edition • Vectorized • Instant Analysis</div>", unsafe_allow_html=True)

    # 1. Load Data
    df, msg, status = load_and_process_data()
    if status == "error": st.error(msg); st.stop()
    
    # 2. Sidebar Control
    with st.sidebar:
        st.markdown("### ⚙️ SCANNER SETTINGS")
        min_score = st.slider("Min Gem Score", 0, 100, 70, 5)
        
        # Filter Sektor
        sectors = ["All"] + sorted(df['Sector'].unique().tolist())
        sel_sector = st.selectbox("Filter Sector", sectors)
        
        if st.button("🔄 REFRESH DATA", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # 3. Main Logic (Instant)
    # Tidak perlu tombol "Start Analysis" lagi karena ini sangat cepat
    results = analyze_gems(df, min_score)
    
    if sel_sector != "All":
        results = results[results['Sector'] == sel_sector]

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["💎 HIDDEN GEMS", "📈 CHART ANALYZER", "📋 RAW DATA"])

    # TAB 1: GEMS
    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("CANDIDATES FOUND", f"{len(results)}")
        if not results.empty:
            c2.metric("TOP PICK", results.iloc[0]['Stock Code'])
            c3.metric("AVG SCORE", f"{results['Gem_Score'].mean():.1f}")
        
        if not results.empty:
            st.markdown("### 🏆 Top Opportunities")
            
            # Format Data for Display
            disp = results[['Stock Code', 'Close', 'Gem_Score', 'SM_3M', 'RSI', 'Volatility', 'Divergence', 'Sector', 'Value']].copy()
            
            st.dataframe(
                disp,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Close": st.column_config.NumberColumn(format="%d"),
                    "Gem_Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f"),
                    "SM_3M": st.column_config.NumberColumn("Smart Money (3M)", format="Rp %.0f"),
                    "RSI": st.column_config.NumberColumn(format="%.1f"),
                    "Volatility": st.column_config.NumberColumn(format="%.2f"),
                    "Value": st.column_config.NumberColumn("Daily Val", format="Rp %.0f")
                }
            )
        else:
            st.warning("No stocks match your criteria. Try lowering the Min Score.")

    # TAB 2: INDIVIDUAL
    with tab2:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        stocks_list = sorted(df['Stock Code'].unique())
        sel_stock = st.selectbox("SELECT STOCK", stocks_list, index=stocks_list.index('BBRI') if 'BBRI' in stocks_list else 0)
        
        if sel_stock:
            df_s = df[df['Stock Code'] == sel_stock].copy()
            
            # Chart
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, 
                               row_heights=[0.5, 0.25, 0.25])
            
            # Price
            fig.add_trace(go.Scatter(x=df_s['Date'], y=df_s['Close'], name='Close', line=dict(color='#2979FF')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_s['Date'], y=df_s['MA20'], name='MA20', line=dict(color='orange', width=1)), row=1, col=1)
            
            # Smart Money (Cumulative)
            df_s['Cum_SM'] = df_s['Smart_Money_Flow'].cumsum()
            fig.add_trace(go.Scatter(x=df_s['Date'], y=df_s['Cum_SM'], name='Smart Money Flow (Cum)', 
                                    line=dict(color='#00E676'), fill='tozeroy'), row=2, col=1)
            
            # RSI
            fig.add_trace(go.Scatter(x=df_s['Date'], y=df_s['RSI'], name='RSI', line=dict(color='#E040FB')), row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="gray", row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="gray", row=3, col=1)
            
            fig.update_layout(height=800, template='plotly_dark', margin=dict(l=10, r=10, t=30, b=10),
                             paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

    # TAB 3: RAW
    with tab3:
        st.dataframe(df.head(100), use_container_width=True)

if __name__ == "__main__":
    main()
