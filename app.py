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

# Custom CSS: Modern Fintech Look
st.markdown("""
<style>
    .stApp { background-color: #F7F9FC; color: #172B4D; }
    div[data-testid="stMetric"] {
        background-color: #FFFFFF; border: 1px solid #E0E0E0;
        padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    h1, h2, h3 { color: #0052CC; font-family: 'Segoe UI', sans-serif; font-weight: 600; }
    div[data-testid="stExpander"] { background-color: #FFFFFF; border: 1px solid #E0E0E0; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. DATA ENGINE
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
        st.error(f"❌ Gagal Autentikasi Google: {e}"); return None

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
# 3. SIDEBAR
# ==============================================================================
st.sidebar.markdown("## 💠 IDX PRO TERMINAL")
st.sidebar.info(f"🟢 **System Online**\n\n📅 Market: {latest_date.date()}\n📅 KSEI: {last_ksei_date.date()}")
st.sidebar.divider()
menu = st.sidebar.radio("Main Navigation", ["🏠 Dashboard Overview", "📊 Stock Analyzer", "🔍 Smart Screener"])

# ==============================================================================
# 4. DASHBOARD (MARKET MAP UPDATE)
# ==============================================================================
if menu == "🏠 Dashboard Overview":
    st.title("Market Pulse")
    st.markdown(f"**Snapshot:** {latest_date.strftime('%A, %d %B %Y')}")
    
    daily_snap = df_daily[df_daily['Last Trading Date'] == latest_date].copy()
    
    # Metrics
    total_val = daily_snap['Value'].sum() / 1e9 
    net_foreign = daily_snap['Net Foreign Flow'].sum() / 1e9
    liquid = daily_snap[daily_snap['Value'] > 1_000_000_000]
    top_gainer = liquid.loc[liquid['Change %'].idxmax()] if not liquid.empty else daily_snap.iloc[0]
    whale_count = daily_snap[daily_snap.get('Big_Player_Anomaly', False) == True].shape[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Value (IDR)", f"{total_val:,.0f} M")
    c2.metric("Net Foreign Flow", f"{net_foreign:,.0f} M", delta_color="normal")
    c3.metric("Top Gainer (Liquid)", f"{top_gainer['Stock Code']}", f"+{top_gainer['Change %']:.1f}%")
    c4.metric("🐋 Whale Radar", f"{whale_count} Alerts")

    st.markdown("---")
    
    # --- [NEW] TABBED VISUALIZATION ---
    tab_map, tab_scatter = st.tabs(["🗺️ Market Map (Treemap)", "📍 Foreign Flow Scatter"])
    
    with tab_map:
        st.subheader("Sektor & Saham Dominan (Finviz Style)")
        # Filter Top 200 by Value agar map rapi
        treemap_data = daily_snap.nlargest(200, 'Value')
        
        # Color Scale Logic (Red to Green)
        fig_tree = px.treemap(
            treemap_data, 
            path=[px.Constant("IHSG"), 'Sector', 'Stock Code'], 
            values='Value',
            color='Change %',
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,
            hover_data=['Close', 'Net Foreign Flow'],
            title=f"Market Map by Transaction Value"
        )
        fig_tree.update_layout(template="plotly_white", margin=dict(t=30, l=10, r=10, b=10), height=600)
        fig_tree.data[0].textinfo = 'label+text+value'
        st.plotly_chart(fig_tree, use_container_width=True)
        st.caption("💡 **Ukuran Kotak** = Nilai Transaksi. **Warna** = Kenaikan/Penurunan Harga.")

    with tab_scatter:
        st.subheader("Foreign Flow vs Price Action")
        top_100 = daily_snap.nlargest(100, 'Value')
        fig_scat = px.scatter(
            top_100, x="Change %", y="Net Foreign Flow", size="Value", color="Sector",
            hover_name="Stock Code", hover_data=["Close", "Avg_Order_Value"], text="Stock Code",
            template="plotly_white", height=600
        )
        fig_scat.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3)
        fig_scat.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.3)
        st.plotly_chart(fig_scat, use_container_width=True)

# ==============================================================================
# 5. STOCK ANALYZER (CUMULATIVE FLOW UPDATE)
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

    # --- [NEW] CUMULATIVE CALCULATION ---
    stock_d['Cum_Foreign'] = stock_d['Net Foreign Flow'].cumsum()
    # Normalisasi agar start dari 0 di grafik
    stock_d['Cum_Foreign'] = stock_d['Cum_Foreign'] - stock_d['Cum_Foreign'].iloc[0]

    # Metrics
    with st.container():
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Close Price", f"Rp {last['Close']:,.0f}", f"{last['Change %']:.2f}%")
        m2.metric("Volume Spike", f"{last['Volume Spike (x)']:.1f}x")
        m3.metric("Bandar Signal", last['Final Signal'])
        m4.metric("Foreign Accumulation (1Y)", f"Rp {stock_d['Net Foreign Flow'].sum()/1e9:,.1f} M", help="Total Net Buy/Sell Asing setahun terakhir")

    st.write("")

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Chart & Accumulation", "⚖️ Peer Comparison", "🏦 KSEI Ownership", "📄 Data"])
    
    with tab1:
        # --- [NEW] HYBRID CHART (Price vs Cumulative Flow) ---
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05,
                            specs=[[{"secondary_y": True}], [{"secondary_y": False}]])
        
        # 1. Candlestick (Primary Y)
        fig.add_trace(go.Candlestick(
            x=stock_d['Last Trading Date'], open=stock_d['Open Price'], 
            high=stock_d['High'], low=stock_d['Low'], close=stock_d['Close'], name='Price'
        ), row=1, col=1, secondary_y=False)
        
        # 2. VWMA (Primary Y)
        fig.add_trace(go.Scatter(x=stock_d['Last Trading Date'], y=stock_d['VWMA_20D'], 
                                 line=dict(color='#0052CC', width=1.5), name='VWMA 20'), row=1, col=1, secondary_y=False)

        # 3. [NEW] Cumulative Foreign Flow (Secondary Y - Line)
        fig.add_trace(go.Scatter(
            x=stock_d['Last Trading Date'], y=stock_d['Cum_Foreign'],
            line=dict(color='#FFAB00', width=2, dash='solid'), name='Cumul. Foreign Flow'
        ), row=1, col=1, secondary_y=True)

        # 4. Volume Bar (Bottom Panel)
        colors = ['#36B37E' if v > 0 else '#FF5630' for v in stock_d['Net Foreign Flow']]
        fig.add_trace(go.Bar(x=stock_d['Last Trading Date'], y=stock_d['Net Foreign Flow'], 
                             marker_color=colors, name='Daily Net Flow'), row=2, col=1)
        
        fig.update_layout(template="plotly_white", height=700, xaxis_rangeslider_visible=False,
                          title=f"Price vs Cumulative Foreign Accumulation: {ticker}", hovermode="x unified")
        fig.update_yaxes(title_text="Price", secondary_y=False, row=1, col=1)
        fig.update_yaxes(title_text="Cumulative Flow (Rp)", secondary_y=True, row=1, col=1, showgrid=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("💡 **Garis Kuning (Kanan)** = Akumulasi Uang Asing. Jika Harga Turun tapi Garis Kuning Naik = **Strong Divergence (Buy)**.")

    with tab2:
        # --- [NEW] PEER COMPARISON ---
        st.subheader(f"Perbandingan {ticker} vs Sektor {last['Sector']}")
        
        # Ambil semua saham di sektor yang sama
        peers = df_daily[(df_daily['Sector'] == last['Sector']) & (df_daily['Last Trading Date'] == latest_date)].copy()
        # Filter saham liquid saja (> 1M transaksi) agar grafik tidak penuh sampah
        peers = peers[peers['Value'] > 500_000_000] 
        
        col_peer1, col_peer2 = st.columns(2)
        
        with col_peer1:
            # Chart 1: Foreign Flow Ranking
            fig_p1 = px.bar(peers.nlargest(10, 'Net Foreign Flow'), 
                            x='Stock Code', y='Net Foreign Flow', color='Net Foreign Flow',
                            color_continuous_scale='RdYlGn', title="Top Foreign Inflow (Sector Peers)")
            st.plotly_chart(fig_p1, use_container_width=True)
            
        with col_peer2:
            # Chart 2: Volume Spike Ranking
            fig_p2 = px.bar(peers.nlargest(10, 'Volume Spike (x)'), 
                            x='Stock Code', y='Volume Spike (x)', 
                            title="Top Volume Spikes (Sector Peers)")
            st.plotly_chart(fig_p2, use_container_width=True)

    with tab3:
        if stock_k.empty:
            st.warning("Data KSEI tidak tersedia.")
        else:
            opts = ['Total_Foreign', 'Local IS', 'Local PF', 'Total_Local']
            sel = st.multiselect("Select Investor:", opts, default=['Total_Foreign', 'Local IS'])
            fig_k = px.line(stock_k, x='Date', y=sel, template="plotly_white", markers=True)
            st.plotly_chart(fig_k, use_container_width=True)

    with tab4:
        st.dataframe(stock_d.sort_values("Last Trading Date", ascending=False), use_container_width=True)

# ==============================================================================
# 6. SMART SCREENER
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
    
    st.info(f"Result: **{len(res)}** stocks found.")
    
    cols = ['Stock Code', 'Close', 'Change %', 'Volume', 'Avg_Order_Value', 'Net Foreign Flow', 'Final Signal']
    st.dataframe(res[[c for c in cols if c in res.columns]].sort_values('Net Foreign Flow', ascending=False), hide_index=True, use_container_width=True)
