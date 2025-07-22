
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import requests
import time
from typing import Dict, List, Tuple, Optional
from itertools import combinations
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import our custom modules
from utils.gsc_processor import GSCProcessor
from utils.serp_scraper import SERPScraper
from utils.overlap_calculator import OverlapCalculator
from utils.report_generator import ReportGenerator

def main():
    st.set_page_config(
        page_title="GSC SERP Overlap Analyzer", 
        page_icon="📊", 
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🔍 GSC Performance & SERP Overlap Analyzer")
    st.markdown("""
    This application analyzes Google Search Console performance data to identify:
    - Query overlap between URLs
    - SERP overlap analysis using Serper API
    - Potential keyword cannibalization issues
    """)

    # Initialize session state
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'serp_data' not in st.session_state:
        st.session_state.serp_data = None
    if 'reports_generated' not in st.session_state:
        st.session_state.reports_generated = False

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")

        # API Key input
        serper_api_key = st.text_input(
            "Serper API Key", 
            type="password",
            help="Get your API key from serper.dev"
        )

        # Upload GSC data
        uploaded_file = st.file_uploader(
            "Upload GSC Performance Report (CSV)", 
            type=['csv'],
            help="Export your GSC performance data with Queries, URLs, Clicks, Impressions, CTR, and Position columns"
        )

        # Analysis parameters
        st.subheader("Analysis Parameters")
        min_clicks = st.number_input(
            "Minimum Clicks for SERP Analysis", 
            min_value=0, 
            value=1,
            help="Only queries with at least this many clicks will be used for SERP overlap analysis"
        )

        serp_overlap_threshold = st.slider(
            "SERP Overlap Threshold (%)", 
            min_value=10, 
            max_value=90, 
            value=50,
            help="Minimum percentage of SERP overlap to flag as potential cannibalization"
        )

        max_queries_to_analyze = st.number_input(
            "Max Queries for SERP Analysis", 
            min_value=10, 
            max_value=1000, 
            value=100,
            help="Limit the number of queries to analyze for SERP overlap (to manage API costs)"
        )

    # Main content area
    if uploaded_file is not None and serper_api_key:
        # Process GSC data
        with st.spinner("Processing GSC data..."):
            processor = GSCProcessor()
            gsc_data = processor.load_and_validate_data(uploaded_file)

            if gsc_data is not None:
                st.success(f"✅ Successfully loaded {len(gsc_data)} rows of GSC data")

                # Display data overview
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total URLs", gsc_data['url'].nunique())
                with col2:
                    st.metric("Total Queries", gsc_data['query'].nunique())
                with col3:
                    st.metric("Total Clicks", f"{gsc_data['clicks'].sum():,}")
                with col4:
                    st.metric("Total Impressions", f"{gsc_data['impressions'].sum():,}")

                # Tabs for different analyses
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🔗 Query Overlap", "🌐 SERP Overlap", "📋 Reports"])

                with tab1:
                    show_data_overview(gsc_data)

                with tab2:
                    show_query_overlap_analysis(gsc_data, processor)

                with tab3:
                    if serper_api_key:
                        show_serp_overlap_analysis(gsc_data, serper_api_key, min_clicks, max_queries_to_analyze, serp_overlap_threshold)
                    else:
                        st.warning("Please enter your Serper API key to perform SERP overlap analysis.")

                with tab4:
                    show_reports(gsc_data, serp_overlap_threshold)
    else:
        st.info("👆 Please upload your GSC data and enter your Serper API key to begin analysis.")

        # Show sample data format
        st.subheader("📋 Expected Data Format")
        sample_data = pd.DataFrame({
            'query': ['python tutorials', 'learn python', 'python guide'],
            'url': ['https://example.com/python-tutorial', 'https://example.com/learn-python', 'https://example.com/python-guide'],
            'clicks': [150, 89, 234],
            'impressions': [2500, 1800, 3200],
            'ctr': [0.06, 0.049, 0.073],
            'position': [3.2, 5.7, 2.1]
        })
        st.dataframe(sample_data)

        st.markdown("""
        **Required Columns:**
        - `query`: Search queries
        - `url`: Landing page URLs  
        - `clicks`: Number of clicks
        - `impressions`: Number of impressions
        - `ctr`: Click-through rate
        - `position`: Average position in search results
        """)

@st.cache_data
def show_data_overview(gsc_data: pd.DataFrame):
    """Display overview of GSC data"""
    st.subheader("📊 Data Overview")

    # Top performing queries
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔥 Top Queries by Clicks")
        top_queries = gsc_data.groupby('query').agg({
            'clicks': 'sum',
            'impressions': 'sum',
            'ctr': 'mean',
            'position': 'mean'
        }).sort_values('clicks', ascending=False).head(10)
        st.dataframe(top_queries.round(3))

    with col2:
        st.subheader("🏆 Top URLs by Clicks")
        top_urls = gsc_data.groupby('url').agg({
            'clicks': 'sum',
            'impressions': 'sum',
            'ctr': 'mean',
            'position': 'mean'
        }).sort_values('clicks', ascending=False).head(10)
        st.dataframe(top_urls.round(3))

def show_query_overlap_analysis(gsc_data: pd.DataFrame, processor: 'GSCProcessor'):
    """Show query overlap analysis between URLs"""
    st.subheader("🔗 Query Overlap Analysis")

    with st.spinner("Calculating query overlaps..."):
        calculator = OverlapCalculator()
        overlap_matrix = calculator.calculate_query_overlap_matrix(gsc_data)

        if overlap_matrix is not None and not overlap_matrix.empty:
            st.success(f"✅ Analyzed {len(overlap_matrix)} URL pairs")

            # Filter significant overlaps
            significant_overlaps = overlap_matrix[overlap_matrix['overlap_percentage'] > 10]

            if not significant_overlaps.empty:
                st.subheader("⚠️ Potential Query Overlaps")
                st.dataframe(significant_overlaps.head(20))

                # Visualization
                fig = px.histogram(
                    overlap_matrix, 
                    x='overlap_percentage', 
                    nbins=20,
                    title="Distribution of Query Overlap Percentages"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No significant query overlaps found.")

def show_serp_overlap_analysis(gsc_data: pd.DataFrame, api_key: str, min_clicks: int, max_queries: int, threshold: int):
    """Show SERP overlap analysis"""
    st.subheader("🌐 SERP Overlap Analysis")

    # Filter queries for SERP analysis
    eligible_queries = gsc_data[gsc_data['clicks'] >= min_clicks]['query'].unique()
    eligible_queries = eligible_queries[:max_queries]  # Limit to manage API costs

    st.info(f"📊 Analyzing {len(eligible_queries)} queries for SERP overlap")

    if st.button("🚀 Start SERP Analysis", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        scraper = SERPScraper(api_key)
        calculator = OverlapCalculator()

        serp_results = {}

        # Scrape SERPs for each query
        for i, query in enumerate(eligible_queries):
            status_text.text(f"Scraping SERP for: {query}")

            serp_data = scraper.get_serp_results(query)
            if serp_data:
                serp_results[query] = serp_data

            progress_bar.progress((i + 1) / len(eligible_queries))
            time.sleep(1)  # Rate limiting

        # Calculate SERP overlaps
        status_text.text("Calculating SERP overlaps...")
        serp_overlaps = calculator.calculate_serp_overlaps(serp_results, gsc_data)

        if serp_overlaps:
            st.session_state.serp_data = serp_overlaps
            st.success(f"✅ Completed SERP analysis for {len(serp_results)} queries")

            # Display results
            display_serp_results(serp_overlaps, threshold)
        else:
            st.error("❌ Failed to analyze SERP overlaps")

    # Display existing results if available
    if st.session_state.serp_data:
        display_serp_results(st.session_state.serp_data, threshold)

def display_serp_results(serp_data: Dict, threshold: int):
    """Display SERP overlap results"""
    # Convert to DataFrame for easier display
    overlap_data = []
    for (query1, query2), overlap_info in serp_data.items():
        overlap_data.append({
            'Query 1': query1,
            'Query 2': query2,
            'SERP Overlap %': overlap_info['overlap_percentage'],
            'Common URLs': len(overlap_info['common_urls']),
            'Total URLs Query 1': len(overlap_info['query1_urls']),
            'Total URLs Query 2': len(overlap_info['query2_urls'])
        })

    if overlap_data:
        df_overlaps = pd.DataFrame(overlap_data)

        # Filter by threshold
        high_overlaps = df_overlaps[df_overlaps['SERP Overlap %'] >= threshold]

        if not high_overlaps.empty:
            st.subheader(f"⚠️ High SERP Overlaps (≥{threshold}%)")
            st.dataframe(high_overlaps.sort_values('SERP Overlap %', ascending=False))

            # Visualization
            fig = px.scatter(
                high_overlaps,
                x='Common URLs',
                y='SERP Overlap %',
                hover_data=['Query 1', 'Query 2'],
                title=f"SERP Overlaps ≥{threshold}%"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No SERP overlaps found above {threshold}% threshold.")

def show_reports(gsc_data: pd.DataFrame, threshold: int):
    """Generate and display reports"""
    st.subheader("📋 Analysis Reports")

    report_generator = ReportGenerator()

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📊 Generate URL-Based Report", type="primary"):
            with st.spinner("Generating URL-based report..."):
                url_report = report_generator.generate_url_report(gsc_data, st.session_state.serp_data, threshold)

                if url_report is not None:
                    st.subheader("🌐 URL-Based Report")
                    st.dataframe(url_report)

                    # Download button
                    csv = url_report.to_csv(index=False)
                    st.download_button(
                        label="📥 Download URL Report",
                        data=csv,
                        file_name=f"url_overlap_report_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )

    with col2:
        if st.button("📝 Generate Query-Based Report", type="primary"):
            with st.spinner("Generating query-based report..."):
                query_report = report_generator.generate_query_report(gsc_data, st.session_state.serp_data)

                if query_report is not None:
                    st.subheader("🔍 Query-Based Report")
                    st.dataframe(query_report)

                    # Download button
                    csv = query_report.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Query Report",
                        data=csv,
                        file_name=f"query_overlap_report_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )

if __name__ == "__main__":
    main()
