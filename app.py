import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# 1. CONFIG & STYLING (Bloomberg Terminal Vibe)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="IDX Institutional Flow Tracker",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk tampilan premium & compact
st.markdown("""
<style>
    /* Main Background & Text */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Metrics Styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.4rem !important;
        color: #00ffca; /* Cyan Bloomberg */
    }
    
    /* Tables */
    .dataframe {
        font-family: 'Consolas', monospace !important; 
    }
    
    /* Remove Padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1f2937;
        border-radius: 5px;
        color: white;
        padding: 10px 20px;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #00ffca;
        color: black;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. HELPER FUNCTIONS & MOCK DATA GENERATOR
# -----------------------------------------------------------------------------

def format_num(num):
    """Format angka dengan pemisah koma (e.g., 1,000,000)."""
    if pd.isna(num):
        return "-"
    return "{:,.0f}".format(num)

def format_pct(num):
    """Format persentase."""
    return "{:,.2f}%".format(num)

@st.cache_data
def generate_mock_ksei_data():
    """
    Simulasi data KSEI (Ownership) yang biasanya Bapak miliki.
    Mencakup data T-1 (Kemarin) dan T (Hari ini) untuk menghitung flow.
    """
    tickers = ['BBCA', 'BBRI', 'TLKM', 'ASII', 'GOTO', 'BUMI']
    data = []
    
    # Simulate data for last 30 days
    start_date = datetime.now() - timedelta(days=30)
    
    for ticker in tickers:
        # Base shares (lembar saham)
        base_shares = np.random.randint(10_000_000, 500_000_000)
        
        # Random distribution logic
        current_date = start_date
        retail_hold = np.random.uniform(0.3, 0.4) # 30-40% Ritel
        
        for _ in range(30):
            # Fluktuasi harian
            change = np.random.uniform(-0.01, 0.01) 
            retail_hold += change
            retail_hold = max(0.1, min(0.9, retail_hold)) # Cap 10-90%
            
            shares_retail = int(base_shares * retail_hold)
            shares_foreign = int(base_shares * (0.3 - (change/2)))
            shares_insurance = int(base_shares * 0.1)
            shares_pension = int(base_shares * 0.1)
            shares_mutual_fund = int(base_shares * 0.05)
            shares_corp = base_shares - (shares_retail + shares_foreign + shares_insurance + shares_pension + shares_mutual_fund)
            
            data.append({
                'Date': current_date.strftime('%Y-%m-%d'),
                'Ticker': ticker,
                'Lokal_Individual': shares_retail, # Ritel (Indikator Distribusi)
                'Asing_Total': shares_foreign,    # Asing (Smart Money)
                'Lokal_Insurance': shares_insurance, # Institutional Long Term
                'Lokal_PensionFund': shares_pension, # Institutional Long Term
                'Lokal_MutualFund': shares_mutual_fund, # Institutional Mid Term
                'Lokal_Corporate': shares_corp,   # Pengendali
                'Total_Shares': base_shares
            })
            current_date += timedelta(days=1)
            
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# -----------------------------------------------------------------------------
# 3. ANALYSIS LOGIC (BANDARMOLOGY)
# -----------------------------------------------------------------------------

def analyze_smart_money_flow(df, ticker):
    """
    Analisa perubahan kepemilikan:
    Jika Ritel Jual & Institusi/Asing Beli -> AKUMULASI (Bagus)
    Jika Ritel Beli & Institusi/Asing Jual -> DISTRIBUSI (Bahaya)
    """
    stock_data = df[df['Ticker'] == ticker].sort_values('Date').reset_index(drop=True)
    
    if len(stock_data) < 2:
        return None, None

    latest = stock_data.iloc[-1]
    prev = stock_data.iloc[-2]
    
    # Hitung Perubahan (Flow)
    flow = {
        'Retail_Flow': latest['Lokal_Individual'] - prev['Lokal_Individual'],
        'Foreign_Flow': latest['Asing_Total'] - prev['Asing_Total'],
        'Inst_Ins_Flow': latest['Lokal_Insurance'] - prev['Lokal_Insurance'],
        'Inst_PF_Flow': latest['Lokal_PensionFund'] - prev['Lokal_PensionFund'],
        'Inst_MF_Flow': latest['Lokal_MutualFund'] - prev['Lokal_MutualFund']
    }
    
    # Hitung Smart Money Net Flow (Asing + Institusi Kuat)
    smart_money_net = flow['Foreign_Flow'] + flow['Inst_Ins_Flow'] + flow['Inst_PF_Flow']
    
    return stock_data, flow, smart_money_net

# -----------------------------------------------------------------------------
# 4. UI COMPONENTS
# -----------------------------------------------------------------------------

def main():
    st.sidebar.title("📊 IDX Data Deck")
    st.sidebar.caption("KSEI Ownership & Bandarmology")
    
    # Load Data
    df_ksei = generate_mock_ksei_data()
    
    # Sidebar Filters
    selected_ticker = st.sidebar.selectbox("Pilih Emiten (Ticker)", df_ksei['Ticker'].unique())
    
    # --- HEADER ---
    st.title(f"{selected_ticker} - Deep Dive Analysis")
    st.markdown("---")

    # Tabs Structure
    tab1, tab2, tab3 = st.tabs(["🏛️ KSEI Bandarmology", "📈 Technical Chart", "📂 Raw Data"])

    # --- TAB 1: KSEI BANDARMOLOGY (NEW FEATURE) ---
    with tab1:
        stock_df, flow, smart_money = analyze_smart_money_flow(df_ksei, selected_ticker)
        
        if stock_df is not None:
            # 1. High Level Summary (Cards)
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Lembar Saham", format_num(stock_df.iloc[-1]['Total_Shares']))
            
            with col2:
                # Retail Change
                delta_retail = flow['Retail_Flow']
                color = "inverse" # Streamlit auto color: Red if pos (bad), Green if neg (good) for retail? 
                # Let's customize via text
                lbl = "Retail (Individu)"
                val = format_num(delta_retail)
                st.metric(lbl, val, delta=format_num(delta_retail), delta_color="inverse")
                
            with col3:
                # Foreign Change
                delta_foreign = flow['Foreign_Flow']
                st.metric("Asing (Foreign)", format_num(delta_foreign), delta=format_num(delta_foreign))
                
            with col4:
                # Smart Money Verdict
                status = "AKUMULASI 🟢" if smart_money > 0 else "DISTRIBUSI 🔴"
                if abs(smart_money) < (stock_df.iloc[-1]['Total_Shares'] * 0.001): status = "NETRAL ⚪"
                
                st.metric("Status Bandar", status, help="Net Flow dari Asing + Asuransi + Dapen")

            st.markdown("---")

            # 2. Detailed Flow Visualization (Waterfall)
            st.subheader("Peta Pergerakan Saham (Flow Map)")
            st.caption(f"Perubahan kepemilikan dari {stock_df.iloc[-2]['Date'].date()} ke {stock_df.iloc[-1]['Date'].date()}")
            
            # Prepare data for Waterfall
            investor_types = ['Ritel (Ind)', 'Asing', 'Asuransi', 'Dapen', 'Reksa Dana']
            flow_values = [
                flow['Retail_Flow'], 
                flow['Foreign_Flow'], 
                flow['Inst_Ins_Flow'], 
                flow['Inst_PF_Flow'], 
                flow['Inst_MF_Flow']
            ]
            
            # Color logic: Green for Buy, Red for Sell
            colors = ['#ef4444' if x < 0 else '#22c55e' for x in flow_values]

            fig_flow = go.Figure(go.Bar(
                x=investor_types,
                y=flow_values,
                marker_color=colors,
                text=[format_num(x) for x in flow_values],
                textposition='auto'
            ))
            
            fig_flow.update_layout(
                title="Net Buy/Sell per Tipe Investor",
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                yaxis_title="Lembar Saham",
                height=400
            )
            st.plotly_chart(fig_flow, use_container_width=True)
            
            # 3. Trend Analysis (Ritel vs Asing)
            st.subheader("Tren Kepemilikan: Ritel vs Smart Money")
            
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(x=stock_df['Date'], y=stock_df['Lokal_Individual'], name='Ritel (Lokal ID)', line=dict(color='yellow', width=2)))
            fig_trend.add_trace(go.Scatter(x=stock_df['Date'], y=stock_df['Asing_Total'], name='Asing', line=dict(color='cyan', width=2)))
            fig_trend.add_trace(go.Scatter(x=stock_df['Date'], y=stock_df['Lokal_Insurance'], name='Asuransi', line=dict(color='purple', width=1, dash='dot')))
            
            fig_trend.update_layout(
                template="plotly_dark",
                xaxis_title="Tanggal",
                yaxis_title="Jumlah Lembar",
                hovermode="x unified",
                height=400,
                legend=dict(orientation="h", y=1.1)
            )
            st.plotly_chart(fig_trend, use_container_width=True)
            
            # Insight Box
            with st.expander("💡 Analisis Insight", expanded=True):
                st.markdown(f"""
                **Logika Analisis:**
                1. Jika Garis **Kuning (Ritel)** turun tajam sementara **Cyan (Asing)** naik, ini sinyal **Strong Buy**.
                2. Perhatikan **Asuransi & Dapen**. Jika mereka keluar (Jual) dalam jumlah besar, biasanya tren jangka panjang berpotensi bearish (mereka jarang *cut loss* kecuali fundamental berubah).
                3. **Net Change hari ini:** Smart Money mencatatkan perubahan sebesar **{format_num(smart_money)}** lembar.
                """)

    # --- TAB 2: TECHNICAL (PLACEHOLDER) ---
    with tab2:
        st.info("Area Analisis Teknikal (Price Action, MA, MACD) akan ditampilkan di sini.")
        # Placeholder chart
        st.line_chart(stock_df.set_index('Date')['Lokal_Individual'])

    # --- TAB 3: RAW DATA ---
    with tab3:
        st.subheader("Data Mentah KSEI")
        
        # Apply formatting to the dataframe for display
        display_df = stock_df.copy()
        numeric_cols = display_df.select_dtypes(include=['number']).columns
        
        # Format for display only (convert to string with commas)
        for col in numeric_cols:
            display_df[col] = display_df[col].apply(lambda x: "{:,.0f}".format(x))
            
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        st.download_button(
            label="Download CSV",
            data=stock_df.to_csv(index=False),
            file_name=f"{selected_ticker}_ksei_data.csv",
            mime='text/csv'
        )

if __name__ == "__main__":
    main()
