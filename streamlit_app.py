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
    cannibalization_threshold: int = 10  # Minimum clicks for cannibalization detection
    click_similarity_ratio: float = 0.5  # URLs must have at least 50% similar clicks for same keyword

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
        
        This tool analyzes Google Search Console data and conducts SERP overlap 
        analysis to identify potential cannibalization issues between your URLs.
        """)
    
    def render_sidebar(self):
        """Render the sidebar with updated layout - API first, then data upload."""
        st.sidebar.title("⚙️ Configuration")
        
        # API Configuration (moved to top as requested)
        st.sidebar.subheader("🔑 API Configuration")
        self._render_api_config()
        
        # File Upload Section (below API config as requested)
        st.sidebar.subheader("📁 Data Upload")
        self._render_file_upload()
        
        # Analysis Parameters
        st.sidebar.subheader("📊 Analysis Parameters")
        self._render_analysis_parameters()
    
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
        """Render analysis parameters section with cannibalization settings."""
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
        
        config.cannibalization_threshold = st.sidebar.number_input(
            "Cannibalization Min Clicks",
            min_value=5,
            max_value=100,
            value=config.cannibalization_threshold,
            help="Minimum clicks required to detect cannibalization"
        )
        
        config.click_similarity_ratio = st.sidebar.slider(
            "Click Similarity Ratio",
            min_value=0.3,
            max_value=1.0,
            value=config.click_similarity_ratio,
            step=0.1,
            help="URLs must have similar click volumes (0.5 = within 50% of each other)"
        )
        
        config.api_delay = st.sidebar.slider(
            "API Delay (seconds)",
            min_value=0.5,
            max_value=5.0,
            value=config.api_delay,
            step=0.5,
            help="Delay between API calls to avoid rate limiting"
        )
        
        # Show actual query count
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
            1. **Enter your Serper API key** in the sidebar
            2. **Upload your GSC data** in the sidebar
            3. **Click Process File**
            4. **Configure analysis parameters**
            5. **Run the analysis**
            """)
        
        with col2:
            st.markdown("### 📊 What You'll Get")
            st.markdown("""
            - **True Cannibalization Detection** based on click volume similarity
            - **SERP Overlap Analysis** using live data
            - **URL-based Reports** for high-level insights
            - **Query-based Reports** for detailed analysis
            """)
        
        st.markdown("### 📋 Cannibalization Detection Logic")
        st.markdown("""
        The app identifies **true cannibalization** when:
        - **Multiple URLs** rank for the same keyword
        - **Both URLs receive substantial clicks** (above minimum threshold)
        - **Click volumes are similar** between URLs (indicating traffic splitting)
        
        **Example**: URL A gets 500 clicks, URL B gets 400 clicks for the same keyword → **Cannibalization detected**
        **Not cannibalization**: URL A gets 21 clicks, URL B gets 1 click → **Proper keyword ownership**
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
                
                query_results = self._analyze_query_overlaps(df, config)
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
    
    def _analyze_query_overlaps(self, df: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
        """Analyze query overlaps with CORRECTED cannibalization detection logic."""
        # Group queries by URL with click data
        url_queries = {}
        for _, row in df.iterrows():
            url = row['url']
            query = row['query']
            clicks = row['clicks']
            
            if url not in url_queries:
                url_queries[url] = {}
            url_queries[url][query] = clicks
        
        # Calculate overlaps with CORRECTED cannibalization logic
        overlaps = []
        cannibalized_queries = []  # Track queries showing true cannibalization
        url_pairs = list(combinations(url_queries.keys(), 2))
        
        for url1, url2 in url_pairs:
            queries1 = set(url_queries[url1].keys())
            queries2 = set(url_queries[url2].keys())
            
            intersection = queries1.intersection(queries2)
            
            if len(intersection) > 0:
                # Traditional overlap percentages (for reference)
                overlap_1_to_2 = len(intersection) / len(queries1) * 100
                overlap_2_to_1 = len(intersection) / len(queries2) * 100
                
                # CORRECTED: Check for actual cannibalization by comparing click volumes
                true_cannibalization_count = 0
                cannibalized_queries_pair = []
                
                for query in intersection:
                    clicks_url1 = url_queries[url1][query]
                    clicks_url2 = url_queries[url2][query]
                    
                    # Only consider queries with sufficient clicks
                    if clicks_url1 >= config.cannibalization_threshold or clicks_url2 >= config.cannibalization_threshold:
                        # Check if clicks are similar (indicating traffic splitting)
                        max_clicks = max(clicks_url1, clicks_url2)
                        min_clicks = min(clicks_url1, clicks_url2)
                        
                        if max_clicks > 0:
                            similarity_ratio = min_clicks / max_clicks
                            
                            # If similarity ratio is above threshold, it's cannibalization
                            if similarity_ratio >= config.click_similarity_ratio:
                                true_cannibalization_count += 1
                                cannibalized_queries_pair.append({
                                    'query': query,
                                    'url1_clicks': clicks_url1,
                                    'url2_clicks': clicks_url2,
                                    'similarity_ratio': similarity_ratio
                                })
                
                # Add to cannibalized queries list
                cannibalized_queries.extend(cannibalized_queries_pair)
                
                # Calculate click shares for traditional metrics
                shared_clicks_url1 = sum(url_queries[url1][query] for query in intersection)
                shared_clicks_url2 = sum(url_queries[url2][query] for query in intersection)
                
                total_clicks_url1 = sum(url_queries[url1].values())
                total_clicks_url2 = sum(url_queries[url2].values())
                
                click_share_url1 = (shared_clicks_url1 / total_clicks_url1) * 100 if total_clicks_url1 > 0 else 0
                click_share_url2 = (shared_clicks_url2 / total_clicks_url2) * 100 if total_clicks_url2 > 0 else 0
                
                overlaps.append({
                    'url1': url1,
                    'url2': url2,
                    'shared_queries': len(intersection),
                    'url1_overlap_pct': round(overlap_1_to_2, 2),
                    'url2_overlap_pct': round(overlap_2_to_1, 2),
                    'url1_click_share_pct': round(click_share_url1, 2),
                    'url2_click_share_pct': round(click_share_url2, 2),
                    'shared_clicks_url1': shared_clicks_url1,
                    'shared_clicks_url2': shared_clicks_url2,
                    'true_cannibalization_count': true_cannibalization_count,
                    'cannibalized_queries': cannibalized_queries_pair,
                    'shared_query_list': list(intersection)
                })
        
        return {
            'url_queries': url_queries,
            'overlaps': overlaps,
            'cannibalized_queries': cannibalized_queries
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
        
        # Cannibalization Report
        if 'query_overlap' in results:
            cannibalization_report = self._create_cannibalization_report(results, df)
            reports['cannibalization'] = cannibalization_report
        
        return reports
    
    def _create_url_based_report(self, df: pd.DataFrame, results: Dict[str, Any], config: AnalysisConfig) -> pd.DataFrame:
        """Create comprehensive URL-based analysis report with CORRECTED cannibalization flags."""
        
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
        
        # Add query overlap information with CORRECTED cannibalization flags
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
        
        # CORRECTED: True cannibalization flags based on click volume similarity
        true_cannibalization_flags = {}
        cannibalization_counts = {}
        
        for overlap in query_overlaps:
            url1, url2 = overlap['url1'], overlap['url2']
            cannibalization_count = overlap['true_cannibalization_count']
            
            if cannibalization_count > 0:
                true_cannibalization_flags[url1] = True
                true_cannibalization_flags[url2] = True
                
                cannibalization_counts[url1] = cannibalization_counts.get(url1, 0) + cannibalization_count
                cannibalization_counts[url2] = cannibalization_counts.get(url2, 0) + cannibalization_count
        
        url_stats['true_cannibalization_flag'] = url_stats.index.map(lambda x: true_cannibalization_flags.get(x, False))
        url_stats['cannibalized_queries_count'] = url_stats.index.map(lambda x: cannibalization_counts.get(x, 0))
        
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
    
    def _create_cannibalization_report(self, results: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
        """Create detailed cannibalization report showing specific query conflicts."""
        cannibalized_queries = results.get('query_overlap', {}).get('cannibalized_queries', [])
        
        if not cannibalized_queries:
            return pd.DataFrame(columns=['query', 'url1', 'url1_clicks', 'url2', 'url2_clicks', 'similarity_ratio', 'total_clicks'])
        
        report_data = []
        for item in cannibalized_queries:
            total_clicks = item['url1_clicks'] + item['url2_clicks']
            
            # Get URL names from the data
            url1_data = df[df['query'] == item['query']].iloc[0]  # Get first match for URL info
            
            report_data.append({
                'query': item['query'],
                'url1': 'URL 1',  # Will be replaced with actual URLs
                'url1_clicks': item['url1_clicks'],
                'url2': 'URL 2',  # Will be replaced with actual URLs  
                'url2_clicks': item['url2_clicks'],
                'similarity_ratio': round(item['similarity_ratio'], 2),
                'total_clicks': total_clicks,
                'click_distribution': f"{item['url1_clicks']}/{item['url2_clicks']}"
            })
        
        cannibalization_df = pd.DataFrame(report_data)
        return cannibalization_df.sort_values('total_clicks', ascending=False)
    
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
        tabs = st.tabs(["📊 Overview", "🌐 URL-Based Report", "🔍 Query-Based Report", "⚠️ Cannibalization Report", "📈 Visualizations"])
        
        with tabs[0]:
            self._render_overview(results)
        
        with tabs[1]:
            self._render_url_report(results)
        
        with tabs[2]:
            self._render_query_report(results)
        
        with tabs[3]:
            self._render_cannibalization_report(results)
        
        with tabs[4]:
            self._render_visualizations(results)
    
    def _render_overview(self, results: Dict[str, Any]):
        """Render comprehensive analysis overview with CORRECTED cannibalization metrics."""
        st.markdown("## 📊 Analysis Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        query_overlaps = results.get('query_overlap', {}).get('overlaps', [])
        serp_overlaps = results.get('serp_overlap', {}).get('overlaps', [])
        cannibalized_queries = results.get('query_overlap', {}).get('cannibalized_queries', [])
        
        with col1:
            st.metric("URL Pairs with Query Overlaps", len(query_overlaps))
        
        with col2:
            true_cannibalization_pairs = len([o for o in query_overlaps if o['true_cannibalization_count'] > 0])
            st.metric("TRUE Cannibalization Cases", true_cannibalization_pairs)
        
        with col3:
            st.metric("Cannibalized Queries", len(cannibalized_queries))
        
        with col4:
            config = st.session_state.analysis_config
            high_serp_overlaps = len([o for o in serp_overlaps if o['serp_overlap_pct'] >= config.serp_overlap_threshold])
            st.metric(f"High SERP Overlaps (≥{config.serp_overlap_threshold}%)", high_serp_overlaps)
        
        # Insights
        st.markdown("### 🎯 Key Insights")
        insights = self._generate_insights(results)
        for insight in insights:
            st.markdown(f"• {insight}")
    
    def _render_cannibalization_report(self, results: Dict[str, Any]):
        """Render detailed cannibalization report."""
        st.markdown("## ⚠️ Cannibalization Report")
        st.markdown("**This report shows queries where multiple URLs receive similar click volumes, indicating true traffic splitting.**")
        
        reports = results.get('reports', {})
        cannibalization_report = reports.get('cannibalization')
        
        if cannibalization_report is not None and not cannibalization_report.empty:
            st.dataframe(cannibalization_report, use_container_width=True, height=400)
            
            # Download button
            csv = cannibalization_report.to_csv(index=False)
            st.download_button(
                label="📥 Download Cannibalization Report",
                data=csv,
                file_name=f"cannibalization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.success("🎉 No true cannibalization detected! Your URLs have proper keyword ownership.")
    
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
        """Render enhanced data visualizations with CORRECTED cannibalization metrics."""
        st.markdown("## 📈 Visualizations")
        
        # True cannibalization distribution
        cannibalized_queries = results.get('query_overlap', {}).get('cannibalized_queries', [])
        if cannibalized_queries:
            similarity_ratios = [item['similarity_ratio'] for item in cannibalized_queries]
            
            fig_cannib = px.histogram(
                x=similarity_ratios,
                nbins=15,
                title="Distribution of Click Similarity Ratios (Cannibalized Queries)",
                labels={'x': 'Click Similarity Ratio', 'y': 'Count of Queries'}
            )
            st.plotly_chart(fig_cannib, use_container_width=True)
        
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
        """Generate actionable insights from analysis results with CORRECTED cannibalization logic."""
        insights = []
        
        query_overlaps = results.get('query_overlap', {}).get('overlaps', [])
        serp_overlaps = results.get('serp_overlap', {}).get('overlaps', [])
        cannibalized_queries = results.get('query_overlap', {}).get('cannibalized_queries', [])
        config = st.session_state.analysis_config
        
        # True cannibalization insights
        if cannibalized_queries:
            insights.append(f"🚨 Found {len(cannibalized_queries)} queries with true cannibalization - multiple URLs receiving similar click volumes")
            
            # Find the most severe cases
            high_traffic_cannibalization = [q for q in cannibalized_queries if q['url1_clicks'] + q['url2_clicks'] >= 50]
            if high_traffic_cannibalization:
                insights.append(f"⚠️ {len(high_traffic_cannibalization)} high-traffic queries are being cannibalized (50+ total clicks)")
                
            # Show similarity ratios
            avg_similarity = sum(q['similarity_ratio'] for q in cannibalized_queries) / len(cannibalized_queries)
            insights.append(f"📊 Average click similarity ratio: {avg_similarity:.2f} (higher = more severe cannibalization)")
        else:
            insights.append("✅ No true cannibalization detected - your URLs have proper keyword ownership")
        
        # SERP overlap insights
        if serp_overlaps:
            high_serp_overlaps = [o for o in serp_overlaps if o['serp_overlap_pct'] >= config.serp_overlap_threshold]
            if high_serp_overlaps:
                insights.append(f"🔍 Identified {len(high_serp_overlaps)} query pairs with high SERP overlap - competing for similar search results")
        
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
