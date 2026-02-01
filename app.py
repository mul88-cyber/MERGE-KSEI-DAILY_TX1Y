import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# ==============================================================================
# 1. KONFIGURASI HALAMAN (MODERN FINTECH LOOK)
# ==============================================================================
st.set_page_config(
    page_title="IDX Pro Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk Dark Mode Profesional ala Bloomberg
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .stMetric { background-color: #1E2129; padding: 10px; border-radius: 8px; border: 1px solid #2E303E; }
    h1, h2, h3 { color: #4DA6FF; font-family: 'Roboto', sans-serif; }
    .metric-label { font-size: 0.8rem; color: #A0A0A0; }
    div[data-testid="stExpander"] { border: 1px solid #41444C; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. DATA LOADER ENGINE
# ==============================================================================
@st.cache_data(ttl=3600)
def load_data():
    try:
        # Load Data Harian
        df_d = pd.read_csv("Kompilasi_Data_1Tahun.csv")
        df_d['Last Trading Date'] = pd.to_datetime(df_d['Last Trading Date'])
        
        # Load Data KSEI (Bulanan)
        df_k = pd.read_csv("KSEI_Shareholder_Processed.csv")
        df_k['Date'] = pd.to_datetime(df_k['Date'])
        
        return df_d, df_k
    except FileNotFoundError:
        st.error("❌ File CSV tidak ditemukan! Pastikan 'Kompilasi_Data_1Tahun.csv' dan 'KSEI_Shareholder_Processed.csv' ada di folder yang sama.")
        return None, None

df_daily, df_ksei = load_data()

if df_daily is None: st.stop()

# Ambil Tanggal Terakhir Data
latest_date = df_daily['Last Trading Date'].max()
last_ksei_date = df_ksei['Date'].max()

# ==============================================================================
# 3. SIDEBAR NAVIGATION
# ==============================================================================
st.sidebar.title("🚀 IDX TERMINAL")
st.sidebar.caption(f"📅 Market Data: {latest_date.date()}")
st.sidebar.caption(f"📅 KSEI Data: {last_ksei_date.date()}")
st.sidebar.divider()

menu = st.sidebar.radio("Main Menu", ["🏠 Market Dashboard", "📊 Stock Analyzer", "🔍 Smart Screener"])

# ==============================================================================
# 4. FITUR: MARKET DASHBOARD
# ==============================================================================
if menu == "🏠 Market Dashboard":
    st.title(f"Market Pulse ({latest_date.strftime('%d %b %Y')})")
    
    # --- SNAPSHOT HARIAN ---
    daily_snap = df_daily[df_daily['Last Trading Date'] == latest_date].copy()
    
    # Hitung Statistik Pasar
    total_val = daily_snap['Value'].sum() / 1e9 # Miliar
    net_foreign = daily_snap['Net Foreign Flow'].sum() / 1e9 # Miliar
    
    # Top Gainer/Loser (Liquid Only > 1M Transaction)
    liquid_stocks = daily_snap[daily_snap['Value'] > 1_000_000_000]
    if liquid_stocks.empty: liquid_stocks = daily_snap # Fallback jika sepi
    
    top_gainer = liquid_stocks.loc[liquid_stocks['Change %'].idxmax()]
    top_loser = liquid_stocks.loc[liquid_stocks['Change %'].idxmin()]
    
    # Big Player Activity (Frequency Anomaly)
    if 'Big_Player_Anomaly' in daily_snap.columns:
        whales_count = daily_snap[daily_snap['Big_Player_Anomaly'] == True].shape[0]
    else:
        whales_count = 0

    # Display Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transaksi (Regular)", f"Rp {total_val:,.0f} M")
    c2.metric("Net Foreign Flow", f"Rp {net_foreign:,.0f} M", delta_color="normal")
    c3.metric("Top Gainer (Liquid)", f"{top_gainer['Stock Code']}", f"+{top_gainer['Change %']:.1f}%")
    c4.metric("🐋 Big Player Alert", f"{whales_count} Emiten", help="Emiten dengan transaksi jumbo tapi frekuensi rendah (Big Order).")

    st.markdown("---")

    # --- CHART: MAP OF MONEY (SCATTER) ---
    st.subheader("🗺️ Map of Money: Foreign Flow vs Price Action")
    
    # Filter Top 100 Value agar chart tidak berat
    top_100 = daily_snap.nlargest(100, 'Value')
    
    fig = px.scatter(
        top_100,
        x="Change %",
        y="Net Foreign Flow",
        size="Value",
        color="Sector",
        hover_name="Stock Code",
        hover_data=["Close", "Signal", "Avg_Order_Value"],
        text="Stock Code",
        title=f"Sektor Mana yang Diakumulasi Asing? (Top 100 Value)",
        template="plotly_dark",
        height=650
    )
    
    # Tambah Garis Kuadran
    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)
    
    # Annotations Kuadran
    fig.add_annotation(x=top_100['Change %'].max(), y=top_100['Net Foreign Flow'].max(), text="STRONG BULLISH (Price Up + Inflow)", showarrow=False, font=dict(color="#00FF00"))
    fig.add_annotation(x=top_100['Change %'].min(), y=top_100['Net Foreign Flow'].min(), text="DUMPING (Price Down + Outflow)", showarrow=False, font=dict(color="#FF0000"))
    
    st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# 5. FITUR: STOCK ANALYZER (DEEP DIVE)
# ==============================================================================
elif menu == "📊 Stock Analyzer":
    st.title("Deep Dive Analysis")
    
    col_sel1, col_sel2 = st.columns([1, 3])
    with col_sel1:
        stock_list = sorted(df_daily['Stock Code'].unique())
        # Default BBCA atau saham pertama
        default_idx = stock_list.index("BBCA") if "BBCA" in stock_list else 0
        ticker = st.selectbox("Pilih Saham:", stock_list, index=default_idx)
    
    # Filter Data Saham Terpilih
    stock_daily = df_daily[df_daily['Stock Code'] == ticker].sort_values("Last Trading Date")
    stock_ksei = df_ksei[df_ksei['Code'] == ticker].sort_values("Date")
    
    # --- METRICS HEADER ---
    last_day = stock_daily.iloc[-1]
    
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Close Price", f"Rp {last_day['Close']:,.0f}", f"{last_day['Change %']:.2f}%")
    m2.metric("Volume Spike", f"{last_day['Volume Spike (x)']:.1f}x Avg", help="Volume hari ini vs Rata-rata 20 hari")
    
    # Signal Logic Color
    sig_color = "normal"
    if "Strong Akumulasi" in last_day['Final Signal']: sig_color = "off" # Greenish trick
    m3.metric("Bandar Signal", last_day['Final Signal'])
    
    # AOV Metric
    aov_val = last_day['Avg_Order_Value'] / 1_000_000 # Dalam Juta
    m4.metric("Avg Order Value", f"Rp {aov_val:,.1f} Jt", help="Rata-rata Nilai per Order. >100 Jt indikasi Institusi.")
    
    m5.metric("Foreign Flow (1D)", f"Rp {last_day['Net Foreign Flow']/1e9:,.1f} M")
    
    # --- TABS ANALYSIS ---
    tab_tech, tab_fund, tab_raw = st.tabs(["📈 Technical & Bandarmology", "🏦 Institutional Flow (KSEI)", "📄 Raw Data"])
    
    with tab_tech:
        # CHART UTAMA: CANDLESTICK + VWMA
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        
        # 1. Candlestick
        fig.add_trace(go.Candlestick(
            x=stock_daily['Last Trading Date'],
            open=stock_daily['Open Price'], high=stock_daily['High'],
            low=stock_daily['Low'], close=stock_daily['Close'],
            name='Price'
        ), row=1, col=1)
        
        # 2. VWMA (Garis Kuning - Indikator Bandar)
        fig.add_trace(go.Scatter(
            x=stock_daily['Last Trading Date'], y=stock_daily['VWMA_20D'],
            line=dict(color='#FFD700', width=2), name='VWMA 20 (Bandar Avg)'
        ), row=1, col=1)
        
        # 3. Volume Bar (Color coded by Foreign Flow)
        colors = ['#00FF00' if v > 0 else '#FF0000' for v in stock_daily['Net Foreign Flow']]
        fig.add_trace(go.Bar(
            x=stock_daily['Last Trading Date'], y=stock_daily['Volume'],
            marker_color=colors, name='Volume (Colored by NFF)'
        ), row=2, col=1)

        # Highlight Big Player Anomaly (Marker Star)
        if 'Big_Player_Anomaly' in stock_daily.columns:
            anomalies = stock_daily[stock_daily['Big_Player_Anomaly'] == True]
            fig.add_trace(go.Scatter(
                x=anomalies['Last Trading Date'], y=anomalies['High']*1.02,
                mode='markers', marker=dict(symbol='star', size=10, color='cyan'),
                name='Big Player Anomaly'
            ), row=1, col=1)

        fig.update_layout(title=f"Chart Analisis {ticker}", template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 **Tips:** Bintang Cyan (⭐) menandakan 'Big Player Anomaly' (Transaksi Jumbo, Frekuensi Rendah). Garis Emas adalah VWMA 20 (Rata-rata harga bandar).")

    with tab_fund:
        if stock_ksei.empty:
            st.warning("⚠️ Data kepemilikan KSEI belum tersedia untuk saham ini.")
        else:
            # CHECK STOCK SPLIT
            is_split = False
            if 'Is_Split_Suspect' in stock_ksei.columns:
                if stock_ksei['Is_Split_Suspect'].tail(3).any():
                    st.error("🚨 PERINGATAN: Terdeteksi potensi STOCK SPLIT baru-baru ini. Lonjakan grafik mungkin bukan akumulasi asli!")
                    is_split = True
            
            st.subheader("Peta Kepemilikan Institusi (KSEI)")
            
            # Pilihan Investor
            inv_options = ['Total_Foreign', 'Local IS', 'Local PF', 'Local MF', 'Foreign IB']
            selected_inv = st.multiselect("Pilih Tipe Investor:", inv_options, default=['Total_Foreign', 'Local IS'])
            
            fig_ksei = px.line(stock_ksei, x='Date', y=selected_inv, markers=True, template="plotly_dark", height=400)
            st.plotly_chart(fig_ksei, use_container_width=True)
            
            # Analisis Flow Bulan Terakhir
            last_ksei = stock_ksei.iloc[-1]
            st.markdown("#### 🕵️‍♂️ Detektif Flow (Data Bulan Terakhir)")
            st.write(f"Periode Data: **{last_ksei['Date'].strftime('%B %Y')}**")
            
            k1, k2 = st.columns(2)
            with k1:
                st.success(f"🟢 Top Buyer: **{last_ksei['Top_Buyer']}**")
                st.metric("Estimasi Beli (Rp)", f"{last_day['Close'] * last_ksei['Top_Buyer_Vol'] / 1e9:,.1f} M") # Estimasi pakai harga closing current
                st.caption(f"Volume: {last_ksei['Top_Buyer_Vol']:,.0f} Lembar")
            with k2:
                st.error(f"🔴 Top Seller: **{last_ksei['Top_Seller']}**")
                st.metric("Estimasi Jual (Rp)", f"{last_day['Close'] * last_ksei['Top_Seller_Vol'] / 1e9:,.1f} M")
                st.caption(f"Volume: {last_ksei['Top_Seller_Vol']:,.0f} Lembar")

    with tab_raw:
        st.dataframe(stock_daily.sort_values("Last Trading Date", ascending=False), use_container_width=True)

# ==============================================================================
# 6. FITUR: SMART SCREENER
# ==============================================================================
elif menu == "🔍 Smart Screener":
    st.title("Pencari Saham Potensial")
    
    # Filter Controls
    with st.expander("🛠️ Filter Settings", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            signal_filter = st.selectbox("Signal Bandarmology", ["All", "Akumulasi", "Strong Akumulasi", "Distribusi"])
        with c2:
            sect_filter = st.selectbox("Sektor", ["All"] + list(df_daily['Sector'].unique()))
        with c3:
            min_aov = st.number_input("Min Avg Order Value (Juta)", value=0, step=10)
        with c4:
            show_whale = st.checkbox("Hanya Show Big Player Anomaly?", value=False)
            
    # Apply Filter
    screener = df_daily[df_daily['Last Trading Date'] == latest_date].copy()
    
    if signal_filter != "All":
        screener = screener[screener['Final Signal'] == signal_filter]
    
    if sect_filter != "All":
        screener = screener[screener['Sector'] == sect_filter]
        
    if min_aov > 0:
        screener = screener[screener['Avg_Order_Value'] >= (min_aov * 1_000_000)]
        
    if show_whale:
        if 'Big_Player_Anomaly' in screener.columns:
            screener = screener[screener['Big_Player_Anomaly'] == True]
            
    # Display Result
    st.success(f"Ditemukan {len(screener)} saham sesuai kriteria.")
    
    cols_display = ['Stock Code', 'Close', 'Change %', 'Volume', 'Avg_Order_Value', 'Net Foreign Flow', 'Final Signal', 'Sector']
    # Filter kolom yang ada saja
    cols_display = [c for c in cols_display if c in screener.columns]
    
    st.dataframe(
        screener[cols_display].sort_values("Net Foreign Flow", ascending=False),
        hide_index=True,
        use_container_width=True,
        column_config={
            "Avg_Order_Value": st.column_config.NumberColumn("Avg Order (Rp)", format="Rp %.0f"),
            "Net Foreign Flow": st.column_config.NumberColumn("Foreign Flow", format="Rp %.0f")
        }
    )
