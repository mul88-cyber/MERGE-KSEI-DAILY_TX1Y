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
# 1. KONFIGURASI HALAMAN & TEMA MODERN (LIGHT MODE PRO)
# ==============================================================================
st.set_page_config(
    page_title="IDX Pro Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS: Modern Fintech Look (White/Grey Theme)
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #F7F9FC;
        color: #172B4D;
    }
    
    /* Card Style untuk Metrics */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Typography Header */
    h1, h2, h3 {
        color: #0052CC; /* Royal Blue */
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 600;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E0E0E0;
    }
    
    /* Expander & Containers */
    div[data-testid="stExpander"] {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 8px;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #FFFFFF;
        border-radius: 5px;
        color: #42526E;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #E3F2FD;
        color: #0052CC;
        border-bottom: 2px solid #0052CC;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. GOOGLE DRIVE CONNECTION ENGINE
# ==============================================================================
FILE_HARIAN = 'Kompilasi_Data_1Tahun.csv'
FILE_KSEI = 'KSEI_Shareholder_Processed.csv'

def get_drive_service():
    try:
        creds_dict = st.secrets["gcp_service_account"]
        creds = service_account.Credentials.from_service_account_info(
            creds_dict, scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        return build('drive', 'v3', credentials=creds, cache_discovery=False)
    except Exception as e:
        st.error(f"❌ Gagal Autentikasi Google: {e}")
        return None

def download_csv_from_drive(service, filename):
    try:
        query = f"name = '{filename}' and trashed = false"
        results = service.files().list(q=query, fields="files(id, name)").execute()
        files = results.get('files', [])
        if not files: return None
        
        request = service.files().get_media(fileId=files[0]['id'])
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False: status, done = downloader.next_chunk()
        fh.seek(0)
        return pd.read_csv(fh)
    except Exception as e:
        st.error(f"❌ Error download '{filename}': {e}"); return None

@st.cache_data(ttl=3600)
def load_data():
    service = get_drive_service()
    if not service: return None, None
    with st.spinner('🔄 Menghubungkan ke IDX Data Lake...'):
        df_d = download_csv_from_drive(service, FILE_HARIAN)
        if df_d is not None: df_d['Last Trading Date'] = pd.to_datetime(df_d['Last Trading Date'])
        df_k = download_csv_from_drive(service, FILE_KSEI)
        if df_k is not None: df_k['Date'] = pd.to_datetime(df_k['Date'])
        return df_d, df_k

df_daily, df_ksei = load_data()
if df_daily is None or df_ksei is None: st.stop()

latest_date = df_daily['Last Trading Date'].max()
last_ksei_date = df_ksei['Date'].max()

# ==============================================================================
# 3. SIDEBAR NAVIGATION
# ==============================================================================
st.sidebar.markdown("## 💠 IDX PRO TERMINAL")
st.sidebar.info(f"🟢 **System Online**\n\n📅 Market: {latest_date.date()}\n📅 KSEI: {last_ksei_date.date()}")
st.sidebar.divider()
menu = st.sidebar.radio("Main Navigation", ["🏠 Dashboard Overview", "📊 Stock Analyzer", "🔍 Smart Screener"])

# ==============================================================================
# 4. DASHBOARD (Light Theme)
# ==============================================================================
if menu == "🏠 Dashboard Overview":
    st.title("Market Pulse")
    st.markdown(f"**Snapshot:** {latest_date.strftime('%A, %d %B %Y')}")
    
    daily_snap = df_daily[df_daily['Last Trading Date'] == latest_date].copy()
    
    # Stats Calculation
    total_val = daily_snap['Value'].sum() / 1e9 
    net_foreign = daily_snap['Net Foreign Flow'].sum() / 1e9
    liquid = daily_snap[daily_snap['Value'] > 1_000_000_000]
    if liquid.empty: liquid = daily_snap
    top_gainer = liquid.loc[liquid['Change %'].idxmax()]
    
    whale_count = daily_snap[daily_snap.get('Big_Player_Anomaly', False) == True].shape[0]

    # Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Value (IDR)", f"{total_val:,.0f} M", help="Total Transaksi Pasar Reguler")
    c2.metric("Net Foreign Flow", f"{net_foreign:,.0f} M", delta_color="normal")
    c3.metric("Top Gainer (Liquid)", f"{top_gainer['Stock Code']}", f"+{top_gainer['Change %']:.1f}%")
    c4.metric("🐋 Whale Radar", f"{whale_count} Alerts", delta="Active", delta_color="off")

    st.markdown("---")
    
    # SCATTER PLOT (WHITE THEME)
    st.subheader("🗺️ Institutional Flow Map")
    top_100 = daily_snap.nlargest(100, 'Value')
    
    fig = px.scatter(
        top_100, x="Change %", y="Net Foreign Flow", size="Value", color="Sector",
        hover_name="Stock Code", hover_data=["Close", "Avg_Order_Value"], text="Stock Code",
        # Ganti template ke 'plotly_white' atau 'plotly' untuk light mode
        template="plotly_white", height=600, title="Top 100 Most Active Stocks"
    )
    # Garis bantu warna abu-abu tipis
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.3)
    
    # Styling text dan marker
    fig.update_traces(textposition='top center', marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# 5. STOCK ANALYZER (Light Theme)
# ==============================================================================
elif menu == "📊 Stock Analyzer":
    st.title("Deep Dive Analysis")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        stock_list = sorted(df_daily['Stock Code'].unique())
        idx_def = stock_list.index("BBCA") if "BBCA" in stock_list else 0
        ticker = st.selectbox("Select Ticker:", stock_list, index=idx_def)
        
    stock_d = df_daily[df_daily['Stock Code'] == ticker].sort_values("Last Trading Date")
    stock_k = df_ksei[df_ksei['Code'] == ticker].sort_values("Date")
    last = stock_d.iloc[-1]

    # Metrics dengan Container Putih
    with st.container():
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Close Price", f"Rp {last['Close']:,.0f}", f"{last['Change %']:.2f}%")
        m2.metric("Volume Spike", f"{last['Volume Spike (x)']:.1f}x")
        m3.metric("Bandar Signal", last['Final Signal'])
        m4.metric("Avg Order Value", f"Rp {last.get('Avg_Order_Value', 0)/1e6:,.0f} Jt")

    st.write("") # Spacing

    tab1, tab2, tab3 = st.tabs(["📈 Chart & Flow", "🏦 KSEI Ownership", "📄 Historical Data"])
    
    with tab1:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
        
        # Price Candle
        fig.add_trace(go.Candlestick(
            x=stock_d['Last Trading Date'], open=stock_d['Open Price'], 
            high=stock_d['High'], low=stock_d['Low'], close=stock_d['Close'], name='Price'
        ), row=1, col=1)
        
        # VWMA (Blue Line for Pro Look)
        fig.add_trace(go.Scatter(x=stock_d['Last Trading Date'], y=stock_d['VWMA_20D'], 
                                 line=dict(color='#0052CC', width=1.5), name='VWMA 20'), row=1, col=1)
        
        # Anomaly Star
        if 'Big_Player_Anomaly' in stock_d.columns:
            anomalies = stock_d[stock_d['Big_Player_Anomaly'] == True]
            fig.add_trace(go.Scatter(x=anomalies['Last Trading Date'], y=anomalies['High']*1.02, 
                                     mode='markers', marker=dict(symbol='star', size=12, color='#FFAB00', line=dict(width=1, color='black')), name='Whale Activity'), row=1, col=1)
        
        # Foreign Flow Bar
        colors = ['#36B37E' if v > 0 else '#FF5630' for v in stock_d['Net Foreign Flow']] # Green/Red Pro colors
        fig.add_trace(go.Bar(x=stock_d['Last Trading Date'], y=stock_d['Net Foreign Flow'], 
                             marker_color=colors, name='Net Foreign Flow'), row=2, col=1)
        
        # Layout Clean White
        fig.update_layout(template="plotly_white", height=650, xaxis_rangeslider_visible=False,
                          hovermode="x unified", title=f"Price Action vs Foreign Flow: {ticker}")
        fig.update_xaxes(showgrid=True, gridcolor='#F4F5F7')
        fig.update_yaxes(showgrid=True, gridcolor='#F4F5F7')
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        if stock_k.empty:
            st.warning("Data KSEI tidak tersedia.")
        else:
            if stock_k.get('Is_Split_Suspect', pd.Series([False])).any():
                st.error("🚨 Potensi Stock Split terdeteksi di histori.")
            
            opts = ['Total_Foreign', 'Local IS', 'Local PF', 'Total_Local']
            sel = st.multiselect("Select Investor Type:", opts, default=['Total_Foreign', 'Local IS'])
            
            # Line Chart KSEI
            fig_k = px.line(stock_k, x='Date', y=sel, template="plotly_white", markers=True, height=500)
            fig_k.update_layout(hovermode="x unified")
            st.plotly_chart(fig_k, use_container_width=True)
            
            # Flow Summary
            lk = stock_k.iloc[-1]
            c_buy, c_sell = st.columns(2)
            with c_buy:
                st.success(f"🟢 **Top Buyer:** {lk['Top_Buyer']}")
                st.caption(f"Accumulated: {lk['Top_Buyer_Vol']:,.0f} shares")
            with c_sell:
                st.error(f"🔴 **Top Seller:** {lk['Top_Seller']}")
                st.caption(f"Distributed: {lk['Top_Seller_Vol']:,.0f} shares")

    with tab3:
        st.dataframe(stock_d.sort_values("Last Trading Date", ascending=False), use_container_width=True)

# ==============================================================================
# 6. SMART SCREENER (Light Theme)
# ==============================================================================
elif menu == "🔍 Smart Screener":
    st.title("Smart Screener")
    
    with st.expander("🛠️  Filter Settings", expanded=True):
        c1, c2, c3 = st.columns(3)
        sig = c1.selectbox("Bandar Signal", ["All", "Akumulasi", "Strong Akumulasi", "Distribusi"])
        sec = c2.selectbox("Sector", ["All"] + list(df_daily['Sector'].unique()))
        whale = c3.checkbox("Show Whale Anomaly Only?")
        
    res = df_daily[df_daily['Last Trading Date'] == latest_date].copy()
    
    if sig != "All": res = res[res['Final Signal'] == sig]
    if sec != "All": res = res[res['Sector'] == sec]
    if whale and 'Big_Player_Anomaly' in res.columns: res = res[res['Big_Player_Anomaly'] == True]
    
    st.info(f"Result: Found **{len(res)}** stocks matching criteria.")
    
    cols = ['Stock Code', 'Close', 'Change %', 'Volume', 'Avg_Order_Value', 'Net Foreign Flow', 'Final Signal']
    st.dataframe(
        res[[c for c in cols if c in res.columns]].sort_values('Net Foreign Flow', ascending=False),
        hide_index=True,
        use_container_width=True
    )
