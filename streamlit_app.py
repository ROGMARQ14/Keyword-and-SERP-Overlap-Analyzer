"""
GSC Performance & SERP Overlap Analyzer
Main Streamlit Application

Enterprise-grade application for analyzing Google Search Console data
and conducting SERP overlap analysis using Serper API.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time
from datetime import datetime
import json
import tempfile
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from itertools import combinations
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import custom utilities
try:
    from utils.gsc_processor import GSCDataProcessor, ColumnMappingError, validate_gsc_file, GSCColumnType
    from utils.serp_scraper import SerpAPIClient, SerpAPIError
    from utils.overlap_calculator import OverlapCalculator, OverlapResult
    from utils.report_generator import ReportGenerator
except ImportError as e:
    st.error(f"Missing utility modules. Please ensure all modules are in the utils/ directory: {e}")
    st.stop()


# Page Configuration
st.set_page_config(
    page_title="GSC Performance & SERP Overlap Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@dataclass
class AnalysisConfig:
    """Configuration for the overlap analysis."""
    min_clicks: int = 1
    serp_overlap_threshold: float = 50.0
    max_queries: int = 100
    api_delay: float = 1.0
    batch_size: int = 10


class StreamlitGSCAnalyzer:
    """Main Streamlit application class for GSC Performance & SERP Overlap Analysis."""
    
    def __init__(self):
        """Initialize the application."""
        self.gsc_processor = GSCDataProcessor()
        self.overlap_calculator = OverlapCalculator()
        self.report_generator = ReportGenerator()
        
        # Initialize session state
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize session state variables."""
        if 'gsc_data' not in st.session_state:
            st.session_state.gsc_data = None
        if 'analysis_config' not in st.session_state:
            st.session_state.analysis_config = AnalysisConfig()
        if 'serp_api_client' not in st.session_state:
            st.session_state.serp_api_client = None
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        if 'analysis_progress' not in st.session_state:
            st.session_state.analysis_progress = 0
    
    def render_header(self):
        """Render the main application header."""
        st.markdown('<h1 class="main-header">🔍 GSC Performance & SERP Overlap Analyzer</h1>', 
                   unsafe_allow_html=True)
        st.markdown("""
        **Identify keyword cannibalization issues through advanced overlap analysis**
        
        This tool analyzes Google Search Console performance data and conducts SERP overlap 
        analysis to identify potential cannibalization issues between your URLs.
        """)
    
    def render_sidebar(self):
        """Render the sidebar with configuration options."""
        st.sidebar.title("⚙️ Configuration")
        
        # File Upload Section
        st.sidebar.subheader("📁 Data Upload")
        self._render_file_upload()
        
        # API Configuration
        st.sidebar.subheader("🔑 API Configuration")
        self._render_api_config()
        
        # Analysis Parameters
        st.sidebar.subheader("📊 Analysis Parameters")
        self._render_analysis_parameters()
        
        # Help Section
        st.sidebar.subheader("❓ Help")
        with st.sidebar.expander("Supported Column Names"):
            self._show_column_mapping_help()
    
    def _render_file_upload(self):
        """Render the file upload section."""
        uploaded_file = st.sidebar.file_uploader(
            "Upload GSC Performance CSV",
            type=['csv'],
            help="Upload your Google Search Console performance data export"
        )
        
        if uploaded_file is not None:
            if st.sidebar.button("Process File"):
                self._process_uploaded_file(uploaded_file)
        
        # Show data summary if loaded
        if st.session_state.gsc_data is not None:
            df = st.session_state.gsc_data
            st.sidebar.success(f"✅ Data loaded: {len(df):,} rows")
            
            with st.sidebar.expander("📊 Data Summary"):
                st.write(f"**URLs:** {df['url'].nunique():,}")
                st.write(f"**Queries:** {df['query'].nunique():,}")
                st.write(f"**Total Clicks:** {df['clicks'].sum():,}")
    
    def _render_api_config(self):
        """Render API configuration section."""
        serp_api_key = st.sidebar.text_input(
            "Serper API Key",
            type="password",
            help="Get your API key from serper.dev",
            value=""
        )
        
        if serp_api_key:
            if st.sidebar.button("Validate API Key"):
                self._validate_api_key(serp_api_key)
            
            # Store API client in session state
            if serp_api_key != "":
                st.session_state.serp_api_client = SerpAPIClient(api_key=serp_api_key)
    
    def _render_analysis_parameters(self):
        """Render analysis parameters section."""
        config = st.session_state.analysis_config
        
        config.min_clicks = st.sidebar.number_input(
            "Minimum Clicks",
            min_value=0,
            max_value=100,
            value=config.min_clicks,
            help="Minimum clicks required for SERP analysis"
        )
        
        config.serp_overlap_threshold = st.sidebar.slider(
            "SERP Overlap Threshold (%)",
            min_value=0.0,
            max_value=100.0,
            value=config.serp_overlap_threshold,
            step=5.0,
            help="Threshold for high SERP overlap classification"
        )
        
        config.max_queries = st.sidebar.number_input(
            "Max Queries to Analyze",
            min_value=10,
            max_value=1000,
            value=config.max_queries,
            step=10,
            help="Maximum number of queries to analyze (cost control)"
        )
        
        config.api_delay = st.sidebar.slider(
            "API Delay (seconds)",
            min_value=0.5,
            max_value=5.0,
            value=config.api_delay,
            step=0.5,
            help="Delay between API calls to avoid rate limiting"
        )
        
        # Cost estimation
        estimated_cost = (config.max_queries / 1000) * 1.0  # $1 per 1000 searches
        st.sidebar.info(f"💰 Estimated cost: ${estimated_cost:.2f}")
    
    def _process_uploaded_file(self, uploaded_file):
        """Process the uploaded GSC CSV file."""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                temp_path = tmp_file.name
            
            # Validate file
            is_valid, message, column_info = validate_gsc_file(temp_path)
            
            if not is_valid:
                st.sidebar.error(f"❌ {message}")
                return
            
            # Process the data
            df = self.gsc_processor.process_gsc_data(temp_path)
            st.session_state.gsc_data = df
            
            # Clean up temp file
            os.unlink(temp_path)
            
            st.sidebar.success("✅ File processed successfully!")
            
        except ColumnMappingError as e:
            st.sidebar.error(f"❌ Column mapping error: {str(e)}")
        except Exception as e:
            st.sidebar.error(f"❌ Processing error: {str(e)}")
    
    def _validate_api_key(self, api_key):
        """Validate the Serper API key."""
        try:
            client = SerpAPIClient(api_key=api_key)
            test_result = client.search("test query", num_results=1)
            
            if test_result:
                st.sidebar.success("✅ API key validated!")
            else:
                st.sidebar.error("❌ API key validation failed")
                
        except Exception as e:
            st.sidebar.error(f"❌ API validation error: {str(e)}")
    
    def _show_column_mapping_help(self):
        """Show column mapping help information."""
        mapping_info = self.gsc_processor.get_column_mapping_info()
        
        for col_type, variations in mapping_info["supported_variations"].items():
            st.write(f"**{col_type.title()}:**")
            st.write(", ".join(variations))
            st.write("")
    
    def render_main_content(self):
        """Render the main content area."""
        if st.session_state.gsc_data is None:
            self._render_welcome_screen()
        else:
            self._render_analysis_interface()
    
    def _render_welcome_screen(self):
        """Render welcome screen when no data is loaded."""
        st.markdown("## 👋 Welcome!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🚀 Getting Started")
            st.markdown("""
            1. **Upload your GSC data** in the sidebar
            2. **Enter your Serper API key**
            3. **Configure analysis parameters**
            4. **Run the analysis**
            """)
        
        with col2:
            st.markdown("### 📊 What You'll Get")
            st.markdown("""
            - **Query Overlap Analysis** between URLs
            - **SERP Overlap Analysis** using live data
            - **URL-based Reports** for high-level insights
            - **Query-based Reports** for detailed analysis
            """)
        
        st.markdown("### 📋 Required Data Format")
        st.markdown("""
        Your GSC CSV export should contain these columns (flexible naming supported):
        - **Query/Queries**: Search terms
        - **URL/Page**: Landing page URLs  
        - **Clicks**: Number of clicks *(required)*
        - **Impressions**: Number of impressions
        - **CTR**: Click-through rate
        - **Position**: Average position
        """)
    
    def _render_analysis_interface(self):
        """Render the main analysis interface."""
        # Analysis controls
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            analysis_type = st.selectbox(
                "Analysis Type",
                ["Full Analysis (Query + SERP Overlap)", "Query Overlap Only", "SERP Overlap Only"],
                help="Choose the type of analysis to perform"
            )
        
        with col2:
            if st.session_state.serp_api_client is None and "SERP" in analysis_type:
                st.error("❌ SERP analysis requires a valid API key")
                run_analysis = False
            else:
                run_analysis = st.button("🚀 Start Analysis", type="primary")
        
        with col3:
            if st.button("🔄 Reset"):
                self._reset_analysis()
        
        # Run analysis
        if run_analysis:
            self._run_analysis(analysis_type)
        
        # Show results if available
        if st.session_state.analysis_results is not None:
            self._render_analysis_results()
    
    def _run_analysis(self, analysis_type: str):
        """Run the selected analysis type."""
        df = st.session_state.gsc_data
        config = st.session_state.analysis_config
        
        # Prepare progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            results = {}
            
            # Query Overlap Analysis
            if "Query" in analysis_type:
                status_text.text("🔄 Running query overlap analysis...")
                progress_bar.progress(25)
                
                query_results = self._analyze_query_overlaps(df)
                results['query_overlap'] = query_results
                
                progress_bar.progress(50)
            
            # SERP Overlap Analysis  
            if "SERP" in analysis_type:
                status_text.text("🔄 Running SERP overlap analysis...")
                
                serp_results = self._analyze_serp_overlaps(df, config, progress_bar, status_text)
                results['serp_overlap'] = serp_results
            
            # Generate Reports
            status_text.text("📊 Generating reports...")
            progress_bar.progress(90)
            
            reports = self._generate_reports(results, config)
            results['reports'] = reports
            
            # Store results
            st.session_state.analysis_results = results
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            # Success message
            st.success("🎉 Analysis completed successfully! Check the results below.")
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            logger.error(f"Analysis error: {e}")
    
    def _analyze_query_overlaps(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze query overlaps between URLs."""
        # Group queries by URL
        url_queries = df.groupby('url')['query'].apply(set).to_dict()
        
        # Calculate overlaps
        overlaps = []
        url_pairs = list(combinations(url_queries.keys(), 2))
        
        for url1, url2 in url_pairs:
            queries1 = url_queries[url1]
            queries2 = url_queries[url2]
            
            intersection = queries1.intersection(queries2)
            
            if len(intersection) > 0:
                overlap_1_to_2 = len(intersection) / len(queries1) * 100
                overlap_2_to_1 = len(intersection) / len(queries2) * 100
                
                overlaps.append({
                    'url1': url1,
                    'url2': url2,
                    'shared_queries': len(intersection),
                    'url1_overlap_pct': overlap_1_to_2,
                    'url2_overlap_pct': overlap_2_to_1,
                    'shared_query_list': list(intersection)
                })
        
        return {
            'url_queries': url_queries,
            'overlaps': overlaps
        }
    
    def _analyze_serp_overlaps(
        self, 
        df: pd.DataFrame, 
        config: AnalysisConfig,
        progress_bar,
        status_text
    ) -> Dict[str, Any]:
        """Analyze SERP overlaps between queries."""
        # Filter queries with minimum clicks
        query_df = df[df['clicks'] >= config.min_clicks].copy()
        
        # Get top queries (limited by config)
        top_queries = query_df.nlargest(config.max_queries, 'clicks')['query'].unique()
        
        # Get SERP data
        serp_data = {}
        api_client = st.session_state.serp_api_client
        
        total_queries = len(top_queries)
        
        for i, query in enumerate(top_queries):
            try:
                status_text.text(f"🔍 Analyzing SERP for: {query[:50]}... ({i+1}/{total_queries})")
                
                # Get SERP results
                serp_results = api_client.search(query, num_results=10)
                
                if serp_results and 'organic' in serp_results:
                    domains = [result.get('link', '') for result in serp_results['organic']]
                    serp_data[query] = domains
                
                # Update progress
                progress = 50 + (i / total_queries) * 40  # 50-90% range
                progress_bar.progress(int(progress))
                
                # Rate limiting
                time.sleep(config.api_delay)
                
            except Exception as e:
                logger.warning(f"SERP analysis failed for query '{query}': {e}")
                continue
        
        # Calculate SERP overlaps
        serp_overlaps = []
        query_pairs = list(combinations(serp_data.keys(), 2))
        
        for query1, query2 in query_pairs:
            if query1 in serp_data and query2 in serp_data:
                serp1 = set(serp_data[query1])
                serp2 = set(serp_data[query2])
                
                intersection = serp1.intersection(serp2)
                union = serp1.union(serp2)
                
                if len(union) > 0:
                    jaccard_similarity = len(intersection) / len(union) * 100
                    
                    if jaccard_similarity >= 10:  # Only include meaningful overlaps
                        serp_overlaps.append({
                            'query1': query1,
                            'query2': query2,
                            'serp_overlap_pct': jaccard_similarity,
                            'shared_domains': list(intersection)
                        })
        
        return {
            'serp_data': serp_data,
            'overlaps': serp_overlaps
        }
    
    def _generate_reports(self, results: Dict[str, Any], config: AnalysisConfig) -> Dict[str, pd.DataFrame]:
        """Generate analysis reports."""
        df = st.session_state.gsc_data
        reports = {}
        
        # URL-Based Report
        if 'query_overlap' in results:
            url_report = self._create_url_based_report(df, results, config)
            reports['url_based'] = url_report
        
        # Query-Based Report
        if 'serp_overlap' in results:
            query_report = self._create_query_based_report(df, results, config)
            reports['query_based'] = query_report
        
        return reports
    
    def _create_url_based_report(self, df: pd.DataFrame, results: Dict[str, Any], config: AnalysisConfig) -> pd.DataFrame:
        """Create URL-based analysis report."""
        url_stats = df.groupby('url').agg({
            'clicks': 'sum',
            'impressions': 'sum',
            'query': 'nunique'
        }).round(2)
        
        url_stats.columns = ['total_clicks', 'total_impressions', 'unique_queries']
        
        # Add query overlap information
        query_overlaps = results.get('query_overlap', {}).get('overlaps', [])
        
        # Calculate high SERP overlaps count
        serp_overlaps = results.get('serp_overlap', {}).get('overlaps', [])
        high_serp_overlaps = {}
        
        for overlap in serp_overlaps:
            if overlap['serp_overlap_pct'] >= config.serp_overlap_threshold:
                # Map queries back to URLs
                for _, row in df.iterrows():
                    if row['query'] in [overlap['query1'], overlap['query2']]:
                        url = row['url']
                        if url not in high_serp_overlaps:
                            high_serp_overlaps[url] = 0
                        high_serp_overlaps[url] += 1
        
        url_stats['high_serp_overlaps'] = url_stats.index.map(lambda x: high_serp_overlaps.get(x, 0))
        
        # Add top queries
        top_queries = df.groupby('url').apply(
            lambda x: ', '.join(x.nlargest(3, 'clicks')['query'].values)
        )
        url_stats['top_queries'] = top_queries
        
        return url_stats.reset_index()
    
    def _create_query_based_report(self, df: pd.DataFrame, results: Dict[str, Any], config: AnalysisConfig) -> pd.DataFrame:
        """Create query-based analysis report."""
        serp_overlaps = results.get('serp_overlap', {}).get('overlaps', [])
        
        query_report_data = []
        
        for overlap in serp_overlaps:
            if overlap['serp_overlap_pct'] >= (config.serp_overlap_threshold / 2):  # Include more data
                query1 = overlap['query1']
                query2 = overlap['query2']
                overlap_pct = overlap['serp_overlap_pct']
                
                # Get URL information for each query
                q1_urls = df[df['query'] == query1]['url'].values
                q2_urls = df[df['query'] == query2]['url'].values
                
                for url in q1_urls:
                    query_report_data.append({
                        'url': url,
                        'query': query1,
                        'serp_overlap_pct': overlap_pct,
                        'competing_query': query2,
                        'competing_urls': ', '.join(q2_urls)
                    })
        
        if query_report_data:
            query_report = pd.DataFrame(query_report_data)
            
            # Add performance metrics
            performance_data = df[['url', 'query', 'clicks', 'impressions', 'ctr', 'position']].copy()
            
            query_report = query_report.merge(
                performance_data,
                on=['url', 'query'],
                how='left'
            )
            
            return query_report.sort_values('serp_overlap_pct', ascending=False)
        else:
            return pd.DataFrame(columns=['url', 'query', 'serp_overlap_pct'])
    
    def _render_analysis_results(self):
        """Render analysis results."""
        results = st.session_state.analysis_results
        
        # Create tabs for different result views
        tabs = st.tabs(["📊 Overview", "🌐 URL-Based Report", "🔍 Query-Based Report", "📈 Visualizations"])
        
        with tabs[0]:
            self._render_overview(results)
        
        with tabs[1]:
            self._render_url_report(results)
        
        with tabs[2]:
            self._render_query_report(results)
        
        with tabs[3]:
            self._render_visualizations(results)
    
    def _render_overview(self, results: Dict[str, Any]):
        """Render analysis overview."""
        st.markdown("## 📊 Analysis Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        query_overlaps = results.get('query_overlap', {}).get('overlaps', [])
        serp_overlaps = results.get('serp_overlap', {}).get('overlaps', [])
        
        with col1:
            st.metric(
                "URL Pairs with Query Overlaps",
                len(query_overlaps)
            )
        
        with col2:
            high_query_overlaps = len([o for o in query_overlaps if o['url1_overlap_pct'] > 50 or o['url2_overlap_pct'] > 50])
            st.metric(
                "High Query Overlaps (>50%)",
                high_query_overlaps
            )
        
        with col3:
            st.metric(
                "Query Pairs with SERP Overlaps",
                len(serp_overlaps)
            )
        
        with col4:
            config = st.session_state.analysis_config
            high_serp_overlaps = len([o for o in serp_overlaps if o['serp_overlap_pct'] >= config.serp_overlap_threshold])
            st.metric(
                f"High SERP Overlaps (≥{config.serp_overlap_threshold}%)",
                high_serp_overlaps
            )
        
        # Insights
        st.markdown("### 🎯 Key Insights")
        
        insights = self._generate_insights(results)
        for insight in insights:
            st.markdown(f"• {insight}")
    
    def _render_url_report(self, results: Dict[str, Any]):
        """Render URL-based report."""
        st.markdown("## 🌐 URL-Based Report")
        
        reports = results.get('reports', {})
        url_report = reports.get('url_based')
        
        if url_report is not None and not url_report.empty:
            st.dataframe(
                url_report,
                use_container_width=True,
                height=400
            )
            
            # Download button
            csv = url_report.to_csv(index=False)
            st.download_button(
                label="📥 Download URL Report",
                data=csv,
                file_name=f"url_based_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No URL-based report data available")
    
    def _render_query_report(self, results: Dict[str, Any]):
        """Render Query-based report."""
        st.markdown("## 🔍 Query-Based Report")
        
        reports = results.get('reports', {})
        query_report = reports.get('query_based')
        
        if query_report is not None and not query_report.empty:
            # Filter controls
            col1, col2 = st.columns(2)
            
            with col1:
                min_overlap = st.slider(
                    "Minimum SERP Overlap %",
                    min_value=0.0,
                    max_value=100.0,
                    value=25.0,
                    step=5.0
                )
            
            with col2:
                max_rows = st.number_input(
                    "Max Rows to Display",
                    min_value=10,
                    max_value=1000,
                    value=100,
                    step=10
                )
            
            # Apply filters
            filtered_report = query_report[
                query_report['serp_overlap_pct'] >= min_overlap
            ].head(max_rows)
            
            st.dataframe(
                filtered_report,
                use_container_width=True,
                height=400
            )
            
            # Download button
            csv = filtered_report.to_csv(index=False)
            st.download_button(
                label="📥 Download Query Report",
                data=csv,
                file_name=f"query_based_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No Query-based report data available")
    
    def _render_visualizations(self, results: Dict[str, Any]):
        """Render data visualizations."""
        st.markdown("## 📈 Visualizations")
        
        # Query Overlap Distribution
        query_overlaps = results.get('query_overlap', {}).get('overlaps', [])
        if query_overlaps:
            overlap_percentages = []
            for overlap in query_overlaps:
                overlap_percentages.extend([overlap['url1_overlap_pct'], overlap['url2_overlap_pct']])
            
            fig_hist = px.histogram(
                x=overlap_percentages,
                nbins=20,
                title="Distribution of Query Overlap Percentages",
                labels={'x': 'Overlap Percentage', 'y': 'Count'}
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # SERP Overlap Distribution
        serp_overlaps = results.get('serp_overlap', {}).get('overlaps', [])
        if serp_overlaps:
            serp_percentages = [overlap['serp_overlap_pct'] for overlap in serp_overlaps]
            
            fig_serp = px.histogram(
                x=serp_percentages,
                nbins=20,
                title="Distribution of SERP Overlap Percentages",
                labels={'x': 'SERP Overlap Percentage', 'y': 'Count'}
            )
            st.plotly_chart(fig_serp, use_container_width=True)
        
        # Top URL Performance
        df = st.session_state.gsc_data
        top_urls = df.groupby('url').agg({
            'clicks': 'sum',
            'impressions': 'sum'
        }).sort_values('clicks', ascending=False).head(10)
        
        fig_urls = px.bar(
            x=top_urls.index,
            y=top_urls['clicks'],
            title="Top 10 URLs by Clicks",
            labels={'x': 'URL', 'y': 'Total Clicks'}
        )
        fig_urls.update_xaxes(tickangle=45)
        st.plotly_chart(fig_urls, use_container_width=True)
    
    def _generate_insights(self, results: Dict[str, Any]) -> List[str]:
        """Generate key insights from analysis results."""
        insights = []
        
        query_overlaps = results.get('query_overlap', {}).get('overlaps', [])
        serp_overlaps = results.get('serp_overlap', {}).get('overlaps', [])
        config = st.session_state.analysis_config
        
        # Query overlap insights
        if query_overlaps:
            high_query_overlaps = [o for o in query_overlaps if o['url1_overlap_pct'] > 80 or o['url2_overlap_pct'] > 80]
            if high_query_overlaps:
                insights.append(f"Found {len(high_query_overlaps)} URL pairs with very high query overlap (>80%) - strong cannibalization indicators")
        
        # SERP overlap insights
        if serp_overlaps:
            high_serp_overlaps = [o for o in serp_overlaps if o['serp_overlap_pct'] >= config.serp_overlap_threshold]
            if high_serp_overlaps:
                insights.append(f"Identified {len(high_serp_overlaps)} query pairs with high SERP overlap - competing for similar search results")
        
        # Performance insights
        df = st.session_state.gsc_data
        zero_click_queries = len(df[df['clicks'] == 0])
        total_queries = len(df)
        
        if zero_click_queries > 0:
            zero_click_pct = (zero_click_queries / total_queries) * 100
            insights.append(f"{zero_click_pct:.1f}% of queries have zero clicks - potential optimization opportunities")
        
        if not insights:
            insights.append("Analysis complete - review the detailed reports for specific optimization opportunities")
        
        return insights
    
    def _reset_analysis(self):
        """Reset analysis results and start over."""
        st.session_state.analysis_results = None
        st.session_state.analysis_progress = 0
        st.success("🔄 Analysis reset. You can now run a new analysis.")
    
    def run(self):
        """Run the main application."""
        self.render_header()
        self.render_sidebar()
        self.render_main_content()


def main():
    """Main application entry point."""
    app = StreamlitGSCAnalyzer()
    app.run()


if __name__ == "__main__":
    main()
