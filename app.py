def main():
    """Main function yang sudah fixed"""
    
    # 1. TAMPILKAN UI KOSONG DULU - CEPAT
    st.set_page_config(layout="wide", page_title="💎 Hidden Gem Finder")
    
    # CSS minimal
    st.markdown("""
    <style>
        .stApp { background: #f8f9fa; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                 padding: 2rem; border-radius: 10px; color: white; margin-bottom: 2rem; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="header">
        <h1>🚀 HIDDEN GEM FINDER v3.0</h1>
        <p>Enterprise Analytics Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. INISIALISASI DI BACKGROUND
    if 'loader' not in st.session_state:
        st.session_state.loader = SimpleDataLoader()
    
    loader = st.session_state.loader
    
    # 3. LOAD DATA DENGAN PROGRESS
    if 'df_merged' not in st.session_state:
        with st.spinner("🔄 Loading data from Google Drive..."):
            try:
                # Load dengan timeout
                import functools
                import threading
                
                @functools.lru_cache(maxsize=1)
                def load_data_once():
                    return loader.load_all_data()
                
                df = load_data_once()
                
                if df.empty:
                    st.error("❌ Failed to load data")
                    st.stop()
                
                st.session_state.df_merged = df
                st.session_state.data_loaded = True
                
                # Success toast
                st.toast("✅ Data loaded successfully!", icon="🎉")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.stop()
    else:
        # Data sudah ada di session state
        df = st.session_state.df_merged
    
    # 4. SIDEBAR (setelah data siap)
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # Quick settings
        lookback = st.slider("Lookback Days", 30, 180, 90)
        min_score = st.slider("Min Score", 50, 90, 70)
        
        if st.button("🔄 Refresh Data", type="primary"):
            st.cache_data.clear()
            st.session_state.clear()
            st.rerun()
    
    # 5. TAB UTAMA (hanya render jika data siap)
    tab1, tab2, tab3 = st.tabs(["🏆 Top Gems", "📈 Analyzer", "📊 Market"])
    
    with tab1:
        # Gunakan analyzer dari session state
        if 'analyzer' not in st.session_state:
            st.session_state.analyzer = EnhancedHiddenGemAnalyzer(df)
        
        analyzer = st.session_state.analyzer
        
        # Tombol untuk run analysis
        if st.button("🔍 Find Hidden Gems", type="primary"):
            with st.spinner(f"Analyzing {min(len(df['Stock Code'].unique()), 200)} stocks..."):
                # SIMPLE VERSION tanpa threading
                results = []
                stocks = df['Stock Code'].unique()[:200]  # Limit 200 saham
                
                for stock in stocks:
                    try:
                        score = analyzer.calculate_enhanced_gem_score(stock, lookback)
                        if score['total_score'] >= min_score:
                            results.append({
                                'Stock': stock,
                                'Score': score['total_score'],
                                'Signal': score['signal'],
                                'Price': score['latest_price'],
                                'SM Flow': score['smart_money_total'] / 1e9
                            })
                    except:
                        pass
                
                if results:
                    results_df = pd.DataFrame(results).sort_values('Score', ascending=False)
                    st.session_state.gem_results = results_df
        
        # Tampilkan hasil jika ada
        if 'gem_results' in st.session_state:
            st.dataframe(st.session_state.gem_results, use_container_width=True)
    
    with tab2:
        st.write("Stock analyzer - coming soon")
    
    with tab3:
        st.write("Market intelligence - coming soon")
