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
# 1. KONFIGURASI HALAMAN & TEMA (LIGHT MODE PRO)
# ==============================================================================
st.set_page_config(
    page_title="IDX Pro Terminal",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Global Theme */
    .stApp { background-color: #F7F9FC; color: #172B4D; }
    
    /* Metrics Card */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF; border: 1px solid #E0E0E0;
        padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetricValue"] {
        font-family: 'Roboto Mono', monospace; font-size: 1.6rem !important;
    }
    
    /* Typography */
    h1, h2, h3 { color: #0052CC; font-family: 'Segoe UI', sans-serif; font-weight: 600; }
    
    /* Tables */
    .dataframe { font-family: 'Consolas', monospace !important; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. DATA ENGINE & FORMATTER
# ==============================================================================
FILE_HARIAN = 'Kompilasi_Data_1Tahun.csv'
FILE_KSEI = 'KSEI_Shareholder_Processed.csv'

# --- HELPER FORMATTING (COMMA SEPARATOR) ---
def fmt_num(val):
    """Format angka standar dengan koma (1,000)"""
    if pd.isna(val): return "-"
    return "{:,.0f}".format(val)

def fmt_idr(val):
    """Format Rupiah Ringkas (M/T) dengan koma"""
    if pd.isna(val): return "-"
    abs_val = abs(val)
    if abs_val >= 1e12: return f"Rp {val/1e12:,.2f} T"
    elif abs_val >= 1e9: return f"Rp {val/1e9:,.2f} M"
    elif abs_val >= 1e6: return f"Rp {val/1e6:,.2f} Jt"
    return f"Rp {val:,.0f}"

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
        # 1. Load Data Harian
        df_d = download_csv_from_drive(service, FILE_HARIAN)
        if df_d is not None: 
            df_d['Last Trading Date'] = pd.to_datetime(df_d['Last Trading Date'])
            # Convert Foreign Flow to Value (IDR)
            if 'Typical Price' in df_d.columns and 'Net Foreign Flow' in df_d.columns:
                df_d['Net Foreign Flow'] = df_d['Net Foreign Flow'] * df_d['Typical Price']
            
        # 2. Load Data KSEI
        df_k = download_csv_from_drive(service, FILE_KSEI)
        if df_k is not None: df_k['Date'] = pd.to_datetime(df_k['Date'])
        
        return df_d, df_k

df_daily, df_ksei = load_data()
if df_daily is None or df_ksei is None: st.stop()

latest_date = df_daily['Last Trading Date'].max()
last_ksei_date = df_ksei['Date'].max()

# --- PRE-CALCULATE ---
df_daily = df_daily.sort_values(['Stock Code', 'Last Trading Date'])

# [FIX] Hitung Avg_Order_Value (Nilai Transaksi per Order)
# Rumus: Value / Frequency
if 'Value' in df_daily.columns and 'Frequency' in df_daily.columns:
    # Hindari pembagian dengan 0
    safe_freq = df_daily['Frequency'].replace(0, 1) 
    df_daily['Avg_Order_Value'] = df_daily['Value'] / safe_freq
else:
    # Fallback jika Value tidak ada (pakai Volume * Close)
    df_daily['Avg_Order_Value'] = 0

# ... kode Flow_1W, Flow_1M Anda yang lama ...
df_daily['Flow_1W'] = df_daily.groupby('Stock Code')['Net Foreign Flow'].transform(lambda x: x.rolling(5, min_periods=1).sum())

# ==============================================================================
# 3. SIDEBAR NAVIGATION
# ==============================================================================
st.sidebar.markdown("## 💠 IDX PRO TERMINAL")
st.sidebar.info(f"🟢 **System Online**\n\n📅 Market: {latest_date.date()}\n📅 KSEI: {last_ksei_date.date()}")
st.sidebar.divider()
menu = st.sidebar.radio("Main Navigation", [
    "🏠 Dashboard Overview", 
    "🏛️ KSEI Intelligence", # NEW TAB
    "📊 Stock Analyzer", 
    "🔍 Smart Screener"
])

# ==============================================================================
# 4. DASHBOARD OVERVIEW
# ==============================================================================
if menu == "🏠 Dashboard Overview":
    st.title("Market Pulse")
    st.markdown(f"**Snapshot:** {latest_date.strftime('%A, %d %B %Y')}")
    
    daily_snap = df_daily[df_daily['Last Trading Date'] == latest_date].copy()
    
    # Metrics
    total_val = daily_snap['Value'].sum()
    net_foreign = daily_snap['Net Foreign Flow'].sum()
    liquid = daily_snap[daily_snap['Value'] > 1_000_000_000]
    top_gainer = liquid.loc[liquid['Change %'].idxmax()] if not liquid.empty else daily_snap.iloc[0]
    whale_count = daily_snap[daily_snap.get('Big_Player_Anomaly', False) == True].shape[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Value (IDR)", fmt_idr(total_val))
    c2.metric("Net Foreign Flow", fmt_idr(net_foreign), delta_color="normal")
    c3.metric("Top Gainer (Liquid)", f"{top_gainer['Stock Code']}", f"+{top_gainer['Change %']:.1f}%")
    c4.metric("🐋 Whale Radar", f"{whale_count} Alerts")

    if whale_count > 0:
        with st.expander(f"🐋 Daftar {whale_count} Saham Whale (Big Player Anomaly)", expanded=False):
            whales = daily_snap[daily_snap.get('Big_Player_Anomaly', False) == True].copy()
            cols_whale = ['Stock Code', 'Close', 'Change %', 'Volume', 'Avg_Order_Value', 'Sector']
            cols_whale = [c for c in cols_whale if c in whales.columns]
            st.dataframe(
                whales[cols_whale].sort_values("Avg_Order_Value", ascending=False),
                hide_index=True, use_container_width=True,
                column_config={
                    "Avg_Order_Value": st.column_config.NumberColumn("Avg Order", format="Rp %.0f"),
                    "Volume": st.column_config.NumberColumn("Volume", format="%.0f"),
                    "Close": st.column_config.NumberColumn("Close", format="Rp %.0f")
                }
            )

    st.markdown("---")
    
    # --- VISUALIZATION TABS ---
    tab_map, tab_scatter = st.tabs(["🗺️ Market Map (Treemap)", "📍 Foreign Flow Scatter"])
    
    with tab_map:
        c_mode1, c_mode2 = st.columns([1, 3])
        with c_mode1:
            map_mode = st.selectbox("Tampilkan Map Berdasarkan:", 
                                    ["Transaction Value", "Foreign Flow (1 Hari)", "Foreign Flow (1 Minggu)", "Foreign Flow (1 Bulan)"])
        
        # --- LOGIC MAP ---
        if map_mode == "Transaction Value":
            treemap_data = daily_snap.nlargest(200, 'Value').copy()
            treemap_data['Value_Text'] = treemap_data['Value'].apply(fmt_idr)
            
            fig_tree = px.treemap(
                treemap_data, path=[px.Constant("IHSG"), 'Sector', 'Stock Code'], 
                values='Value', color='Change %',
                color_continuous_scale=['#D32F2F', '#E0E0E0', '#00C853'], range_color=[-3, 3],
                custom_data=['Value_Text', 'Close', 'Change %'], title="Market Map by Transaction Value"
            )
            fig_tree.update_traces(
                texttemplate="<b>%{label}</b><br>%{customdata[0]}<br>%{customdata[2]:.2f}%",
                hovertemplate="<b>%{label}</b><br>Val: %{customdata[0]}<br>Price: %{customdata[1]:,.0f}<br>Chg: %{customdata[2]:.2f}%"
            )
            
        else: # FOREIGN FLOW (VALUE) MODES
            if "1 Hari" in map_mode: col_target = 'Net Foreign Flow'
            elif "1 Minggu" in map_mode: col_target = 'Flow_1W'
            else: col_target = 'Flow_1M'
            
            treemap_data = daily_snap.copy()
            treemap_data['Abs_Flow'] = treemap_data[col_target].abs()
            treemap_data = treemap_data.nlargest(200, 'Abs_Flow')
            treemap_data['Flow_Text'] = treemap_data[col_target].apply(fmt_idr)
            
            fig_tree = px.treemap(
                treemap_data, path=[px.Constant("IHSG"), 'Sector', 'Stock Code'], 
                values='Abs_Flow', color=col_target, 
                color_continuous_scale=['#D32F2F', '#E0E0E0', '#00C853'],
                color_continuous_midpoint=0,
                custom_data=['Flow_Text', 'Close', 'Change %'],
                title=f"Foreign Flow Map (Value): {map_mode}"
            )
            fig_tree.update_traces(
                texttemplate="<b>%{label}</b><br>%{customdata[0]}",
                hovertemplate="<b>%{label}</b><br>Net Flow: %{customdata[0]}<br>Price: %{customdata[1]:,.0f}<br>Chg: %{customdata[2]:.2f}%"
            )

        fig_tree.update_layout(template="plotly_white", margin=dict(t=30, l=10, r=10, b=10), height=600)
        st.plotly_chart(fig_tree, use_container_width=True)

    with tab_scatter:
        st.subheader("Foreign Flow vs Price Action (Top Movers)")
        top_100 = daily_snap.nlargest(100, 'Value').copy()
        top_100['Label'] = np.where(
            (top_100['Value'] >= top_100['Value'].nlargest(20).min()) |
            (top_100['Change %'].abs() >= top_100['Change %'].abs().nlargest(5).min()), 
            top_100['Stock Code'], ""
        )
        
        fig_scat = px.scatter(
            top_100, x="Change %", y="Net Foreign Flow", size="Value", color="Sector",
            hover_name="Stock Code", hover_data=["Close", "Avg_Order_Value"], text="Label",
            template="plotly_white", height=600
        )
        fig_scat.update_traces(textposition='top center', textfont=dict(size=10, color='black'))
        fig_scat.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3)
        fig_scat.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.3)
        st.plotly_chart(fig_scat, use_container_width=True)

# ==============================================================================
# 5. [NEW] KSEI INTELLIGENCE (DEEP DIVE ANALYTICS)
# ==============================================================================
elif menu == "🏛️ KSEI Intelligence":
    st.title("KSEI Smart Money Tracker")
    st.markdown("Analisis pergerakan kepemilikan saham: **Siapa yang menampung saat Ritel membuang barang?**")
    
    # 1. Prepare Data Logic
    # Kita butuh pivot data KSEI: Tanggal Terakhir vs Tanggal Sebelumnya (misal 30 hari lalu)
    
    # Ambil list saham yang ada di KSEI
    ksei_codes = df_ksei['Code'].unique()
    
    # Filter 2 Tanggal untuk perbandingan
    dates = sorted(df_ksei['Date'].unique())
    if len(dates) < 2:
        st.warning("Data KSEI tidak cukup untuk analisis historis (kurang dari 2 periode).")
        st.stop()
        
    col_d1, col_d2 = st.columns(2)
    date_end = col_d1.selectbox("Tanggal Akhir (Current):", dates, index=len(dates)-1)
    # Default compare to T-1 (Previous Data Point)
    date_start_idx = max(0, len(dates)-2) 
    date_start = col_d2.selectbox("Tanggal Awal (Previous):", dates, index=date_start_idx)
    
    if date_start >= date_end:
        st.error("Tanggal Awal harus lebih kecil dari Tanggal Akhir.")
        st.stop()
        
    # Process Logic
    with st.spinner("Menganalisis jutaan baris data KSEI..."):
        df_end = df_ksei[df_ksei['Date'] == date_end].set_index('Code')
        df_start = df_ksei[df_ksei['Date'] == date_start].set_index('Code')
        
        # Saham yang ada di kedua periode
        common_codes = df_end.index.intersection(df_start.index)
        
        analysis = pd.DataFrame(index=common_codes)
        
        # Calculate Flow (Lembar)
        analysis['Retail_Flow'] = df_end.loc[common_codes, 'Local ID'] - df_start.loc[common_codes, 'Local ID']
        analysis['Foreign_Flow'] = df_end.loc[common_codes, 'Total_Foreign'] - df_start.loc[common_codes, 'Total_Foreign']
        analysis['Local_Inst_Flow'] = (df_end.loc[common_codes, 'Local IS'] + df_end.loc[common_codes, 'Local PF']) - \
                                      (df_start.loc[common_codes, 'Local IS'] + df_start.loc[common_codes, 'Local PF'])
        
        analysis['Price_End'] = df_end.loc[common_codes, 'Price']
        analysis['Price_Start'] = df_start.loc[common_codes, 'Price']
        
        # Price Change %
        analysis['Price_Chg_Pct'] = ((analysis['Price_End'] - analysis['Price_Start']) / analysis['Price_Start']) * 100
        analysis['Value_Traded_Est'] = abs(analysis['Retail_Flow']) * analysis['Price_End'] # Estimasi nilai barang yg pindah
        
        # LOGIC: Ritel Buang Barang (-) = Good Signal
        analysis['Retail_Action'] = np.where(analysis['Retail_Flow'] < 0, "Distribusi (Good)", "Akumulasi (Bad)")
        
        # Merge Sector
        analysis = analysis.reset_index().rename(columns={'Code': 'Stock Code'})
        sector_map = df_daily[['Stock Code', 'Sector']].drop_duplicates().set_index('Stock Code')
        analysis = analysis.merge(sector_map, on='Stock Code', how='left')

    # 2. METRICS
    st.markdown("---")
    # Top Accumulation (Retail Out, Price Stable/Up)
    # Filter: Price Chg > -2% (Tidak crash), Retail Flow Negative
    candidates = analysis[(analysis['Retail_Flow'] < 0) & (analysis['Price_Chg_Pct'] > -5)]
    top_acc = candidates.sort_values('Value_Traded_Est', ascending=False).head(1)
    
    # Top Distribution (Retail In, Price Down/Up)
    dist_candidates = analysis[analysis['Retail_Flow'] > 0]
    top_dist = dist_candidates.sort_values('Value_Traded_Est', ascending=False).head(1)
    
    m1, m2, m3 = st.columns(3)
    if not top_acc.empty:
        tkr = top_acc.iloc[0]['Stock Code']
        val = top_acc.iloc[0]['Retail_Flow']
        m1.metric("Top Akumulasi (Ritel Kabur)", tkr, f"{fmt_num(val)} Lbr", delta_color="inverse")
        
    if not top_dist.empty:
        tkr = top_dist.iloc[0]['Stock Code']
        val = top_dist.iloc[0]['Retail_Flow']
        m2.metric("Top Distribusi (Ritel Nampung)", tkr, f"+{fmt_num(val)} Lbr", delta_color="inverse")
        
    m3.metric("Total Emiten Teranalisis", f"{len(analysis)} Saham")
    
    # 3. QUADRANT CHART
    st.subheader("Peta 'Bandarmology': Retail Flow vs Price Action")
    st.caption("💡 **Area Emas (Kiri Atas):** Harga Naik + Ritel Jual (Akumulasi Smart Money).")
    
    # Filter outliers for better chart
    chart_data = analysis[analysis['Value_Traded_Est'] > 1_000_000_000] # Min Value 1M biar chart ga penuh saham tidur
    
    fig_quad = px.scatter(
        chart_data, 
        x="Retail_Flow", 
        y="Price_Chg_Pct", 
        size="Value_Traded_Est", 
        color="Sector",
        hover_name="Stock Code",
        hover_data=["Foreign_Flow", "Local_Inst_Flow", "Price_End"],
        text="Stock Code",
        title=f"Scatter Plot: Perubahan Ritel vs Perubahan Harga ({date_start.date()} - {date_end.date()})"
    )
    
    # Add Quadrant Lines
    fig_quad.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_quad.add_vline(x=0, line_dash="dot", line_color="gray")
    
    # Annotations
    fig_quad.add_annotation(x=chart_data['Retail_Flow'].min(), y=chart_data['Price_Chg_Pct'].max(),
                            text="🔥 AKUMULASI (Strong)", showarrow=False, font=dict(color="green", size=14))
    fig_quad.add_annotation(x=chart_data['Retail_Flow'].max(), y=chart_data['Price_Chg_Pct'].min(),
                            text="⚠️ DISTRIBUSI (Weak)", showarrow=False, font=dict(color="red", size=14))

    fig_quad.update_traces(textposition='top center')
    fig_quad.update_layout(height=650, template="plotly_white")
    st.plotly_chart(fig_quad, use_container_width=True)

    # 4. DATA TABLE
    st.subheader("Data Detail Aliran Dana KSEI")
    
    # Format Table
    show_cols = ['Stock Code', 'Sector', 'Price_End', 'Price_Chg_Pct', 'Retail_Flow', 'Foreign_Flow', 'Local_Inst_Flow']
    
    st.dataframe(
        analysis[show_cols].sort_values('Retail_Flow'), # Sort by retail exit (most negative first)
        hide_index=True,
        use_container_width=True,
        column_config={
            "Price_End": st.column_config.NumberColumn("Harga", format="Rp %.0f"),
            "Price_Chg_Pct": st.column_config.NumberColumn("Chg %", format="%.2f%%"),
            "Retail_Flow": st.column_config.NumberColumn("Ritel Flow (Lbr)", format="%.0f"),
            "Foreign_Flow": st.column_config.NumberColumn("Asing Flow (Lbr)", format="%.0f"),
            "Local_Inst_Flow": st.column_config.NumberColumn("Inst Lokal Flow (Lbr)", format="%.0f"),
        }
    )

# ==============================================================================
# 6. STOCK ANALYZER
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

    # Cumulative Flow
    stock_d['Cum_Foreign'] = stock_d['Net Foreign Flow'].cumsum()
    stock_d['Cum_Foreign'] = stock_d['Cum_Foreign'] - stock_d['Cum_Foreign'].iloc[0]

    with st.container():
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Close Price", f"Rp {fmt_num(last['Close'])}", f"{last['Change %']:.2f}%")
        m2.metric("Volume Spike", f"{last['Volume Spike (x)']:.1f}x")
        m3.metric("Bandar Signal", last['Final Signal'])
        m4.metric("Foreign Accum (1Y)", fmt_idr(stock_d['Net Foreign Flow'].sum()), help="Total Net Flow Asing (Value) Setahun")

    st.write("")
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Chart & Accumulation", "⚖️ Peer Comparison", "🏦 KSEI Ownership", "📄 Data"])
    
    with tab1:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05,
                            specs=[[{"secondary_y": True}], [{"secondary_y": False}]])
        
        # Price Candle
        fig.add_trace(go.Candlestick(x=stock_d['Last Trading Date'], open=stock_d['Open Price'], high=stock_d['High'], low=stock_d['Low'], close=stock_d['Close'], name='Price'), row=1, col=1, secondary_y=False)
        # VWMA
        fig.add_trace(go.Scatter(x=stock_d['Last Trading Date'], y=stock_d['VWMA_20D'], line=dict(color='#0052CC', width=1.5), name='VWMA 20'), row=1, col=1, secondary_y=False)
        # Cumulative Flow (Value)
        fig.add_trace(go.Scatter(x=stock_d['Last Trading Date'], y=stock_d['Cum_Foreign'], line=dict(color='#FFAB00', width=2), name='Cumul. Foreign Flow (Rp)'), row=1, col=1, secondary_y=True)
        
        # Whale Star
        if 'Big_Player_Anomaly' in stock_d.columns:
            anomalies = stock_d[stock_d['Big_Player_Anomaly'] == True]
            fig.add_trace(go.Scatter(x=anomalies['Last Trading Date'], y=anomalies['High']*1.05, mode='markers', marker=dict(symbol='star', size=14, color='#FFD700', line=dict(width=1, color='black')), name='Whale Activity'), row=1, col=1, secondary_y=False)
        
        # Daily Net Flow Bar (Value)
        colors = ['#36B37E' if v > 0 else '#FF5630' for v in stock_d['Net Foreign Flow']]
        fig.add_trace(go.Bar(x=stock_d['Last Trading Date'], y=stock_d['Net Foreign Flow'], marker_color=colors, name='Daily Net Flow (Rp)'), row=2, col=1)
        
        fig.update_layout(template="plotly_white", height=700, xaxis_rangeslider_visible=False, title=f"Price vs Cumulative Foreign Flow (Value): {ticker}", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader(f"Perbandingan {ticker} vs Sektor {last['Sector']}")
        peers = df_daily[(df_daily['Sector'] == last['Sector']) & (df_daily['Last Trading Date'] == latest_date)].copy()
        peers = peers[peers['Value'] > 500_000_000] 
        c_p1, c_p2 = st.columns(2)
        with c_p1:
            fig_p1 = px.bar(peers.nlargest(10, 'Net Foreign Flow'), x='Stock Code', y='Net Foreign Flow', color='Net Foreign Flow', color_continuous_scale='RdYlGn', title="Top Foreign Inflow (Value)")
            st.plotly_chart(fig_p1, use_container_width=True)
        with c_p2:
            fig_p2 = px.bar(peers.nlargest(10, 'Volume Spike (x)'), x='Stock Code', y='Volume Spike (x)', title="Top Volume Spikes")
            st.plotly_chart(fig_p2, use_container_width=True)

    with tab3:
        if stock_k.empty:
            st.warning("Data KSEI tidak tersedia.")
        else:
            # Multi-select untuk data KSEI
            opts = ['Total_Foreign', 'Local IS', 'Local PF', 'Local ID', 'Local MF']
            sel = st.multiselect("Pilih Tipe Investor:", opts, default=['Total_Foreign', 'Local ID'])
            
            # Line Chart Sederhana
            fig_k = px.line(stock_k, x='Date', y=sel, markers=True, title=f"Tren Kepemilikan {ticker}")
            fig_k.update_layout(template="plotly_white", hovermode="x unified")
            st.plotly_chart(fig_k, use_container_width=True)
            
            # Flow Analysis Table (Last vs Prev)
            if len(stock_k) >= 2:
                last_row = stock_k.iloc[-1]
                prev_row = stock_k.iloc[-2]
                st.write(f"**Perubahan Terakhir ({prev_row['Date'].date()} ke {last_row['Date'].date()}):**")
                
                cols_check = ['Local ID', 'Total_Foreign', 'Local IS', 'Local PF']
                diffs = {}
                for c in cols_check:
                    diffs[c] = last_row[c] - prev_row[c]
                
                d_df = pd.DataFrame([diffs]).T.rename(columns={0: 'Net Change (Lembar)'})
                d_df['Status'] = d_df['Net Change (Lembar)'].apply(lambda x: "🟢 Akumulasi" if x > 0 else "🔴 Distribusi")
                 
                # Cek nilai Ritel secara manual karena ini scalar (single value)
                retail_val = d_df.loc['Local ID', 'Net Change (Lembar)']
                
                # Update status khusus row 'Local ID'
                if retail_val < 0:
                    d_df.loc['Local ID', 'Status'] = "🔴 Panik Jual (Good)"
                else:
                    d_df.loc['Local ID', 'Status'] = "🟢 Ritel Masuk (Bad)"
                
                st.table(d_df)

    with tab4:
        st.dataframe(stock_d.sort_values("Last Trading Date", ascending=False), use_container_width=True)

# ==============================================================================
# 7. SMART SCREENER
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
    
    # Format Table Output
    cols = ['Stock Code', 'Close', 'Change %', 'Volume', 'Avg_Order_Value', 'Net Foreign Flow', 'Final Signal']
    st.dataframe(
        res[[c for c in cols if c in res.columns]].sort_values('Net Foreign Flow', ascending=False), 
        hide_index=True, use_container_width=True,
        column_config={
            "Net Foreign Flow": st.column_config.NumberColumn("Foreign Flow (Rp)", format="Rp %.0f"),
            "Avg_Order_Value": st.column_config.NumberColumn("Avg Order (Rp)", format="Rp %.0f"),
            "Volume": st.column_config.NumberColumn("Volume", format="%.0f"),
            "Close": st.column_config.NumberColumn("Close", format="Rp %.0f"),
        }
    )
