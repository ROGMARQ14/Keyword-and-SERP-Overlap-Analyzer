"""
GSC Performance & SERP Overlap Analyzer
Complete Streamlit Application with Enhanced CSV Processing

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
import csv
import chardet
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from itertools import combinations
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    api_delay: float = 1.0
    batch_size: int = 10
    # Removed max_queries - no artificial limits per user request

class EnhancedCSVProcessor:
    """Enhanced CSV processor with automatic delimiter detection and flexible column mapping."""
    
    def __init__(self):
        self.common_delimiters = [',', ';', '\t', '|', ':']
        self.column_variations = {
            'query': ["query", "queries", "keyword", "keywords", "keyqords", 
                     "search term", "search terms", "search query", "search queries"],
            'url': ["url", "urls", "page", "pages", "landing page", "landing pages",
                   "address", "addresses", "destination url", "destination page"],
            'clicks': ["click", "clicks", "total clicks", "click count"],
            'impressions': ["impression", "impressions", "total impressions", "impression count"],
            'ctr': ["ctr", "url ctr", "click through rate", "click-through rate",
                   "clickthrough rate", "click thru rate"],
            'position': ["position", "positions", "avg pos", "avg. pos", "avg position",
                        "average position", "avg. position", "ranking", "rank"]
        }
    
    def detect_delimiter(self, file_path: str, sample_size: int = 8192) -> str:
        """Detect CSV delimiter using multiple methods with fallbacks."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                sample = f.read(sample_size)
                
            if sample:
                sniffer = csv.Sniffer()
                try:
                    dialect = sniffer.sniff(sample, delimiters=''.join(self.common_delimiters))
                    logger.info(f"CSV Sniffer detected delimiter: '{dialect.delimiter}'")
                    return dialect.delimiter
                except csv.Error:
                    logger.warning("CSV Sniffer failed, trying frequency analysis")
            
            # Frequency analysis fallback
            delimiter = self._frequency_based_detection(sample)
            if delimiter:
                logger.info(f"Frequency analysis detected delimiter: '{delimiter}'")
                return delimiter
                
        except Exception as e:
            logger.error(f"Delimiter detection error: {e}")
        
        # Default fallback
        logger.warning("Using default comma delimiter")
        return ','
    
    def _frequency_based_detection(self, sample: str) -> Optional[str]:
        """Detect delimiter based on character frequency in first few lines."""
        if not sample:
            return None
            
        lines = sample.split('\n')[:5]
        delimiter_counts = {delim: 0 for delim in self.common_delimiters}
        
        for line in lines:
            if line.strip():
                for delim in self.common_delimiters:
                    delimiter_counts[delim] += line.count(delim)
        
        max_count = max(delimiter_counts.values())
        if max_count > 0:
            for delim, count in delimiter_counts.items():
                if count == max_count:
                    return delim
        
        return None
    
    def detect_encoding(self, file_path: str, sample_size: int = 8192) -> str:
        """Detect file encoding using chardet."""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(sample_size)
            
            result = chardet.detect(raw_data)
            encoding = result.get('encoding', 'utf-8')
            confidence = result.get('confidence', 0)
            
            logger.info(f"Encoding detection: {encoding} (confidence: {confidence:.2f})")
            
            if confidence < 0.7:
                for fallback_encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        with open(file_path, 'r', encoding=fallback_encoding) as f:
                            f.read(1024)
                        logger.info(f"Using fallback encoding: {fallback_encoding}")
                        return fallback_encoding
                    except UnicodeDecodeError:
                        continue
            
            return encoding
            
        except Exception as e:
            logger.warning(f"Encoding detection failed: {e}, using utf-8")
            return 'utf-8'
    
    def create_column_mapping(self, csv_columns: List[str]) -> Dict[str, str]:
        """Create flexible column mapping from CSV columns to standard names."""
        column_mapping = {}
        mapped_types = set()
        
        # Normalize column names
        normalized_columns = {
            col.lower().strip().replace('_', ' '): col 
            for col in csv_columns
        }
        
        # Try to match each column type
        for col_type, variations in self.column_variations.items():
            matched_column = None
            
            for variation in variations:
                normalized_variation = variation.lower().strip()
                
                # Direct match
                if normalized_variation in normalized_columns:
                    matched_column = normalized_columns[normalized_variation]
                    break
                
                # Partial match
                for norm_csv_col, original_csv_col in normalized_columns.items():
                    if (normalized_variation in norm_csv_col or 
                        norm_csv_col in normalized_variation):
                        matched_column = original_csv_col
                        break
                
                if matched_column:
                    break
            
            if matched_column:
                column_mapping[matched_column] = col_type
                mapped_types.add(col_type)
                logger.debug(f"Mapped: {matched_column} -> {col_type}")
        
        return column_mapping
    
    def process_gsc_data(self, file_path: str) -> pd.DataFrame:
        """Process GSC CSV data with automatic delimiter detection and column mapping."""
        try:
            # Step 1: Detect delimiter and encoding
            delimiter = self.detect_delimiter(file_path)
            encoding = self.detect_encoding(file_path)
            
            # Step 2: Load CSV
            df = pd.read_csv(
                file_path,
                sep=delimiter,
                encoding=encoding,
                engine='python',
                on_bad_lines='warn'
            )
            
            logger.info(f"Loaded CSV with shape: {df.shape}")
            logger.info(f"Original columns: {list(df.columns)}")
            
            # Step 3: Create column mapping
            column_mapping = self.create_column_mapping(df.columns)
            
            # Check if required columns are mapped
            required_cols = ['query', 'url', 'clicks']
            missing_required = set(required_cols) - set(column_mapping.values())
            
            if missing_required:
                available_variations = []
                for missing_col in missing_required:
                    if missing_col in self.column_variations:
                        variations = self.column_variations[missing_col]
                        available_variations.append(f"{missing_col}: {', '.join(variations)}")
                
                error_msg = (
                    f"Missing required columns: {list(missing_required)}. "
                    f"Available columns: {list(df.columns)}. "
                    f"\nSupported column name variations:\n" + 
                    '\n'.join(available_variations)
                )
                raise ValueError(error_msg)
            
            # Step 4: Apply mapping and clean data
            df_mapped = df[list(column_mapping.keys())].rename(columns=column_mapping)
            
            # Data cleaning
            df_clean = df_mapped.dropna(subset=required_cols)
            
            # Convert numeric columns
            numeric_columns = ['clicks', 'impressions', 'ctr', 'position']
            for col in numeric_columns:
                if col in df_clean.columns:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
            # Remove rows with 0 clicks
            if 'clicks' in df_clean.columns:
                initial_count = len(df_clean)
                df_clean = df_clean[df_clean['clicks'] > 0]
                removed_count = initial_count - len(df_clean)
                
                if removed_count > 0:
                    logger.info(f"Removed {removed_count} rows with 0 or negative clicks")
            
            # Clean text columns
            text_columns = ['url', 'query']
            for col in text_columns:
                if col in df_clean.columns:
                    df_clean[col] = df_clean[col].astype(str).str.strip()
                    df_clean = df_clean[df_clean[col] != '']
            
            if len(df_clean) == 0:
                raise ValueError("No valid data remaining after cleaning and validation")
            
            logger.info(f"Data processing complete. Final dataset: {len(df_clean):,} rows")
            return df_clean
            
        except Exception as e:
            error_msg = f"Failed to process GSC data: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

class SerpAPIClient:
    """SERP API client for search result analysis."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://google.serper.dev/search"
    
    def search(self, query: str, num_results: int = 10) -> Dict[str, Any]:
        """Perform search and return results."""
        try:
            headers = {
                'X-API-KEY': self.api_key,
                'Content-Type': 'application/json'
            }
            
            payload = {
                'q': query,
                'num': num_results
            }
            
            response = requests.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"SERP API error for query '{query}': {e}")
            return {}

class StreamlitGSCAnalyzer:
    """Main Streamlit application class for GSC Performance & SERP Overlap Analysis."""
    
    def __init__(self):
        """Initialize the application."""
        self.csv_processor = EnhancedCSVProcessor()
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
        """Render the enhanced file upload section."""
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
            
            if serp_api_key != "":
                st.session_state.serp_api_client = SerpAPIClient(api_key=serp_api_key)
    
    def _render_analysis_parameters(self):
        """Render analysis parameters section - NO query limits per user request."""
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
        
        config.api_delay = st.sidebar.slider(
            "API Delay (seconds)",
            min_value=0.5,
            max_value=5.0,
            value=config.api_delay,
            step=0.5,
            help="Delay between API calls to avoid rate limiting"
        )
        
        # Show actual query count instead of artificial limits
        if st.session_state.gsc_data is not None:
            df = st.session_state.gsc_data
            eligible_queries = len(df[df['clicks'] >= config.min_clicks])
            st.sidebar.info(f"📊 Queries to analyze: {eligible_queries:,}")
    
    def _process_uploaded_file(self, uploaded_file):
        """Process the uploaded GSC CSV file with enhanced error handling."""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                temp_path = tmp_file.name
            
            # Process the data using enhanced CSV processor
            df = self.csv_processor.process_gsc_data(temp_path)
            st.session_state.gsc_data = df
            
            # Clean up temp file
            os.unlink(temp_path)
            
            st.sidebar.success("✅ File processed successfully!")
            
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
        """Show enhanced column mapping help information."""
        mapping_info = {
            "Query/Keywords": "query, queries, keyword, keywords, search term",
            "URL/Page": "url, page, landing page, address, destination url",
            "Clicks": "click, clicks, total clicks",
            "Impressions": "impression, impressions, total impressions",
            "CTR": "ctr, url ctr, click through rate",
            "Position": "position, avg pos, average position, rank"
        }
        
        for col_type, variations in mapping_info.items():
            st.write(f"**{col_type}:**")
            st.write(variations)
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
        - **Query/Queries**: Search terms *(required)*
        - **URL/Page**: Landing page URLs *(required)*
        - **Clicks**: Number of clicks *(required)*
        - **Impressions**: Number of impressions *(optional)*
        - **CTR**: Click-through rate *(optional)*
        - **Position**: Average position *(optional)*
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
        """Run the selected analysis type with enhanced progress tracking."""
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
    
    def _analyze_serp_overlaps(self, df: pd.DataFrame, config: AnalysisConfig, progress_bar, status_text) -> Dict[str, Any]:
        """Analyze SERP overlaps between queries with NO artificial limits."""
        # Filter queries with minimum clicks
        query_df = df[df['clicks'] >= config.min_clicks].copy()
        
        # Get ALL unique queries (no limit) - sort by clicks for better progress tracking
        top_queries = query_df.sort_values('clicks', ascending=False)['query'].unique()
        
        # Get SERP data
        serp_data = {}
        api_client = st.session_state.serp_api_client
        
        total_queries = len(top_queries)
        st.info(f"🔍 Analyzing SERP data for {total_queries:,} queries. This may take several minutes...")
        
        for i, query in enumerate(top_queries):
            try:
                status_text.text(f"🔍 Analyzing SERP for: {query[:50]}... ({i+1:,}/{total_queries:,})")
                
                # Get SERP results
                serp_results = api_client.search(query, num_results=10)
                
                if serp_results and 'organic' in serp_results:
                    domains = [result.get('link', '') for result in serp_results['organic']]
                    serp_data[query] = domains
                
                # Update progress
                progress = 50 + (i / total_queries) * 40
                progress_bar.progress(int(progress))
                
                # Rate limiting
                time.sleep(config.api_delay)
                
            except Exception as e:
                logger.warning(f"SERP analysis failed for query '{query}': {e}")
                continue
        
        # Calculate SERP overlaps using Jaccard similarity
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
                    
                    if jaccard_similarity >= 10:
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
        """Generate comprehensive analysis reports."""
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
        """Create comprehensive URL-based analysis report with optional column support."""
        
        # Build aggregation dictionary based on available columns
        agg_dict = {'query': 'nunique'}  # Always available
        
        if 'clicks' in df.columns:
            agg_dict['clicks'] = 'sum'
        if 'impressions' in df.columns:
            agg_dict['impressions'] = 'sum'
        
        url_stats = df.groupby('url').agg(agg_dict).round(2)
        
        # Rename columns for clarity
        column_rename = {'query': 'unique_queries'}
        if 'clicks' in url_stats.columns:
            column_rename['clicks'] = 'total_clicks'
        if 'impressions' in url_stats.columns:
            column_rename['impressions'] = 'total_impressions'
        
        url_stats = url_stats.rename(columns=column_rename)
        
        # Add query overlap information
        query_overlaps = results.get('query_overlap', {}).get('overlaps', [])
        
        # Calculate high SERP overlaps count
        serp_overlaps = results.get('serp_overlap', {}).get('overlaps', [])
        high_serp_overlaps = {}
        
        for overlap in serp_overlaps:
            if overlap['serp_overlap_pct'] >= config.serp_overlap_threshold:
                for _, row in df.iterrows():
                    if row['query'] in [overlap['query1'], overlap['query2']]:
                        url = row['url']
                        if url not in high_serp_overlaps:
                            high_serp_overlaps[url] = 0
                        high_serp_overlaps[url] += 1
        
        url_stats['high_serp_overlaps'] = url_stats.index.map(lambda x: high_serp_overlaps.get(x, 0))
        
        # Add top queries - Fix pandas warning
        if 'clicks' in df.columns:
            top_queries = (
                df.groupby('url', group_keys=False)[['query', 'clicks']]
                .apply(lambda g: ', '.join(g.nlargest(3, 'clicks')['query'].values), include_groups=False)
            )
        else:
            # Fallback if no clicks column
            top_queries = (
                df.groupby('url', group_keys=False)['query']
                .apply(lambda g: ', '.join(g.head(3).values), include_groups=False)
            )
        
        url_stats['top_queries'] = top_queries
        
        return url_stats.reset_index()
    
    def _create_query_based_report(self, df: pd.DataFrame, results: Dict[str, Any], config: AnalysisConfig) -> pd.DataFrame:
        """Create detailed query-based analysis report with defensive column handling."""
        serp_overlaps = results.get('serp_overlap', {}).get('overlaps', [])
        
        query_report_data = []
        
        for overlap in serp_overlaps:
            if overlap['serp_overlap_pct'] >= (config.serp_overlap_threshold / 2):
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
            
            # Add performance metrics - ONLY include columns that actually exist
            available_perf_cols = ['url', 'query']  # Always required
            optional_cols = ['clicks', 'impressions', 'ctr', 'position']
            
            for col in optional_cols:
                if col in df.columns:
                    available_perf_cols.append(col)
            
            performance_data = df[available_perf_cols].copy()
            
            query_report = query_report.merge(
                performance_data,
                on=['url', 'query'],
                how='left'
            )
            
            return query_report.sort_values('serp_overlap_pct', ascending=False)
        else:
            return pd.DataFrame(columns=['url', 'query', 'serp_overlap_pct'])
    
    def _render_analysis_results(self):
        """Render comprehensive analysis results with enhanced visualizations."""
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
        """Render comprehensive analysis overview."""
        st.markdown("## 📊 Analysis Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        query_overlaps = results.get('query_overlap', {}).get('overlaps', [])
        serp_overlaps = results.get('serp_overlap', {}).get('overlaps', [])
        
        with col1:
            st.metric("URL Pairs with Query Overlaps", len(query_overlaps))
        
        with col2:
            high_query_overlaps = len([o for o in query_overlaps if o['url1_overlap_pct'] > 50 or o['url2_overlap_pct'] > 50])
            st.metric("High Query Overlaps (>50%)", high_query_overlaps)
        
        with col3:
            st.metric("Query Pairs with SERP Overlaps", len(serp_overlaps))
        
        with col4:
            config = st.session_state.analysis_config
            high_serp_overlaps = len([o for o in serp_overlaps if o['serp_overlap_pct'] >= config.serp_overlap_threshold])
            st.metric(f"High SERP Overlaps (≥{config.serp_overlap_threshold}%)", high_serp_overlaps)
        
        # Insights
        st.markdown("### 🎯 Key Insights")
        insights = self._generate_insights(results)
        for insight in insights:
            st.markdown(f"• {insight}")
    
    def _render_url_report(self, results: Dict[str, Any]):
        """Render URL-based report with download functionality."""
        st.markdown("## 🌐 URL-Based Report")
        
        reports = results.get('reports', {})
        url_report = reports.get('url_based')
        
        if url_report is not None and not url_report.empty:
            st.dataframe(url_report, use_container_width=True, height=400)
            
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
        """Render Query-based report with interactive filtering."""
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
            
            st.dataframe(filtered_report, use_container_width=True, height=400)
            
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
        """Render enhanced data visualizations."""
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
        if 'clicks' in df.columns:
            top_urls = df.groupby('url').agg({
                'clicks': 'sum',
                'impressions': 'sum' if 'impressions' in df.columns else lambda x: 0
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
        """Generate actionable insights from analysis results."""
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
        if 'clicks' in df.columns:
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
        """Run the main application with comprehensive error handling."""
        try:
            self.render_header()
            self.render_sidebar()
            self.render_main_content()
        except Exception as e:
            st.error(f"Application error: {str(e)}")
            logger.error(f"Application error: {e}")

def main():
    """Main application entry point with error handling."""
    try:
        app = StreamlitGSCAnalyzer()
        app.run()
    except Exception as e:
        st.error(f"Failed to initialize application: {str(e)}")
        logger.error(f"Initialization error: {e}")

if __name__ == "__main__":
    main()
