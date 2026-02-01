import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import io
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ==============================================================================
# 1. KONFIGURASI HALAMAN
# ==============================================================================
st.set_page_config(
    page_title="IDX Pro Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Dark Mode
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .stMetric { background-color: #1E2129; padding: 10px; border-radius: 8px; border: 1px solid #2E303E; }
    h1, h2, h3 { color: #4DA6FF; font-family: 'Roboto', sans-serif; }
    div[data-testid="stExpander"] { border: 1px solid #41444C; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. GOOGLE DRIVE CONNECTION ENGINE
# ==============================================================================

# Nama File yang dicari di Google Drive (Harus persis)
FILE_HARIAN = 'Kompilasi_Data_1Tahun.csv'
FILE_KSEI = 'KSEI_Shareholder_Processed.csv'

def get_drive_service():
    """Membuat koneksi ke GDrive menggunakan Service Account dari st.secrets"""
    try:
        # Load credentials dari Streamlit Secrets
        creds_dict = st.secrets["gcp_service_account"]
        creds = service_account.Credentials.from_service_account_info(
            creds_dict,
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        return build('drive', 'v3', credentials=creds, cache_discovery=False)
    except Exception as e:
        st.error(f"❌ Gagal Autentikasi Google: {e}")
        return None

def download_csv_from_drive(service, filename):
    """Mencari file berdasarkan nama, download ke memory, convert ke DataFrame"""
    try:
        # 1. Cari File ID berdasarkan Nama
        query = f"name = '{filename}' and trashed = false"
        results = service.files().list(q=query, fields="files(id, name)").execute()
        files = results.get('files', [])

        if not files:
            st.warning(f"⚠️ File '{filename}' tidak ditemukan di Google Drive Service Account.")
            return None

        file_id = files[0]['id']
        
        # 2. Download Konten
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        
        # 3. Baca CSV dari Memory
        fh.seek(0)
        return pd.read_csv(fh)
        
    except Exception as e:
        st.error(f"❌ Error download '{filename}': {e}")
        return None

@st.cache_data(ttl=3600) # Cache data 1 jam agar hemat kuota API
def load_data():
    service = get_drive_service()
    if not service: return None, None
    
    with st.spinner('Menghubungkan ke IDX Database (Google Drive)...'):
        # Load Data Harian
        df_d = download_csv_from_drive(service, FILE_HARIAN)
        if df_d is not None:
            df_d['Last Trading Date'] = pd.to_datetime(df_d['Last Trading Date'])

        # Load Data KSEI
        df_k = download_csv_from_drive(service, FILE_KSEI)
        if df_k is not None:
            df_k['Date'] = pd.to_datetime(df_k['Date'])
            
        return df_d, df_k

# --- LOAD DATA START ---
df_daily, df_ksei = load_data()

if df_daily is None or df_ksei is None:
    st.error("Gagal memuat data. Pastikan Service Account sudah dijadikan 'Editor/Viewer' di Folder Google Drive.")
    st.stop()

# Meta Data
latest_date = df_daily['Last Trading Date'].max()
last_ksei_date = df_ksei['Date'].max()

# ==============================================================================
# 3. SIDEBAR NAVIGATION
# ==============================================================================
st.sidebar.title("🚀 IDX TERMINAL")
st.sidebar.success("🟢 Online & Connected")
st.sidebar.caption(f"📅 Market Data: {latest_date.date()}")
st.sidebar.caption(f"📅 KSEI Data: {last_ksei_date.date()}")
st.sidebar.divider()

menu = st.sidebar.radio("Main Menu", ["🏠 Market Dashboard", "📊 Stock Analyzer", "🔍 Smart Screener"])

# ==============================================================================
# 4. FITUR: MARKET DASHBOARD
# ==============================================================================
if menu == "🏠 Market Dashboard":
    st.title(f"Market Pulse ({latest_date.strftime('%d %b %Y')})")
    
    daily_snap = df_daily[df_daily['Last Trading Date'] == latest_date].copy()
    
    # Stats
    total_val = daily_snap['Value'].sum() / 1e9 
    net_foreign = daily_snap['Net Foreign Flow'].sum() / 1e9
    
    # Top Movers (Liquid > 1M)
    liquid = daily_snap[daily_snap['Value'] > 1_000_000_000]
    if liquid.empty: liquid = daily_snap
    
    top_gainer = liquid.loc[liquid['Change %'].idxmax()]
    top_loser = liquid.loc[liquid['Change %'].idxmin()]
    
    # Whale Alert
    whale_count = daily_snap[daily_snap.get('Big_Player_Anomaly', False) == True].shape[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transaksi", f"Rp {total_val:,.0f} M")
    c2.metric("Net Foreign Flow", f"Rp {net_foreign:,.0f} M")
    c3.metric("Top Gainer", f"{top_gainer['Stock Code']}", f"+{top_gainer['Change %']:.1f}%")
    c4.metric("🐋 Whale Alert", f"{whale_count} Emiten")

    st.markdown("---")
    
    # MAP OF MONEY
    st.subheader("🗺️ Map of Money: Foreign Flow vs Price Action")
    top_100 = daily_snap.nlargest(100, 'Value')
    
    fig = px.scatter(
        top_100, x="Change %", y="Net Foreign Flow", size="Value", color="Sector",
        hover_name="Stock Code", hover_data=["Close", "Avg_Order_Value"], text="Stock Code",
        template="plotly_dark", height=600, title="Top 100 Value Stocks"
    )
    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)
    fig.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.3)
    st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# 5. FITUR: STOCK ANALYZER
# ==============================================================================
elif menu == "📊 Stock Analyzer":
    st.title("Deep Dive Analysis")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        stock_list = sorted(df_daily['Stock Code'].unique())
        idx_def = stock_list.index("BBCA") if "BBCA" in stock_list else 0
        ticker = st.selectbox("Pilih Saham:", stock_list, index=idx_def)
        
    stock_d = df_daily[df_daily['Stock Code'] == ticker].sort_values("Last Trading Date")
    stock_k = df_ksei[df_ksei['Code'] == ticker].sort_values("Date")
    last = stock_d.iloc[-1]

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Close", f"Rp {last['Close']:,.0f}", f"{last['Change %']:.2f}%")
    m2.metric("Volume Spike", f"{last['Volume Spike (x)']:.1f}x")
    m3.metric("Bandar Signal", last['Final Signal'])
    m4.metric("Avg Order Value", f"Rp {last.get('Avg_Order_Value', 0)/1e6:,.0f} Jt")

    tab1, tab2, tab3 = st.tabs(["📈 Technical & Bandar", "🏦 Institutional Flow (KSEI)", "📄 Raw Data"])
    
    with tab1:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
        # Price
        fig.add_trace(go.Candlestick(
            x=stock_d['Last Trading Date'], open=stock_d['Open Price'], 
            high=stock_d['High'], low=stock_d['Low'], close=stock_d['Close'], name='Price'
        ), row=1, col=1)
        # VWMA
        fig.add_trace(go.Scatter(x=stock_d['Last Trading Date'], y=stock_d['VWMA_20D'], 
                                 line=dict(color='#FFD700', width=1.5), name='VWMA 20'), row=1, col=1)
        # Anomaly
        if 'Big_Player_Anomaly' in stock_d.columns:
            anomalies = stock_d[stock_d['Big_Player_Anomaly'] == True]
            fig.add_trace(go.Scatter(x=anomalies['Last Trading Date'], y=anomalies['High']*1.02, 
                                     mode='markers', marker=dict(symbol='star', size=10, color='cyan'), name='Whale Activity'), row=1, col=1)
        # Net Foreign
        colors = ['#00FF00' if v > 0 else '#FF0000' for v in stock_d['Net Foreign Flow']]
        fig.add_trace(go.Bar(x=stock_d['Last Trading Date'], y=stock_d['Net Foreign Flow'], 
                             marker_color=colors, name='Net Foreign Flow'), row=2, col=1)
        
        fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        if stock_k.empty:
            st.warning("Data KSEI tidak tersedia.")
        else:
            if stock_k.get('Is_Split_Suspect', pd.Series([False])).any():
                st.error("🚨 Potensi Stock Split terdeteksi di histori.")
            
            opts = ['Total_Foreign', 'Local IS', 'Local PF', 'Total_Local']
            sel = st.multiselect("Pilih Investor:", opts, default=['Total_Foreign', 'Local IS'])
            fig_k = px.line(stock_k, x='Date', y=sel, template="plotly_dark", markers=True)
            st.plotly_chart(fig_k, use_container_width=True)
            
            # Flow Analysis Last Month
            lk = stock_k.iloc[-1]
            st.info(f"Top Buyer Bulan {lk['Date'].strftime('%B')}: **{lk['Top_Buyer']}** (Vol: {lk['Top_Buyer_Vol']:,.0f})")

    with tab3:
        st.dataframe(stock_d.sort_values("Last Trading Date", ascending=False), use_container_width=True)

# ==============================================================================
# 6. FITUR: SMART SCREENER
# ==============================================================================
elif menu == "🔍 Smart Screener":
    st.title("Smart Screener")
    
    with st.expander("Filter Options", expanded=True):
        c1, c2, c3 = st.columns(3)
        sig = c1.selectbox("Signal", ["All", "Akumulasi", "Strong Akumulasi", "Distribusi"])
        sec = c2.selectbox("Sector", ["All"] + list(df_daily['Sector'].unique()))
        whale = c3.checkbox("Show Whale Anomaly Only?")
        
    res = df_daily[df_daily['Last Trading Date'] == latest_date].copy()
    
    if sig != "All": res = res[res['Final Signal'] == sig]
    if sec != "All": res = res[res['Sector'] == sec]
    if whale and 'Big_Player_Anomaly' in res.columns: res = res[res['Big_Player_Anomaly'] == True]
    
    st.success(f"Result: {len(res)} stocks")
    cols = ['Stock Code', 'Close', 'Change %', 'Volume', 'Avg_Order_Value', 'Net Foreign Flow', 'Final Signal']
    st.dataframe(res[[c for c in cols if c in res.columns]].sort_values('Net Foreign Flow', ascending=False), hide_index=True, use_container_width=True)
