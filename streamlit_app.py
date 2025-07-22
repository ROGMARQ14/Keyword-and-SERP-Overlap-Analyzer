"""
GSC Keyword Cannibalization Analyzer
Enterprise-grade application for detecting true keyword cannibalization issues

Focuses specifically on identifying cases where multiple URLs are splitting traffic
for the same queries, indicating genuine cannibalization problems.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile
import os
import csv
import chardet
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from itertools import combinations
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page Configuration
st.set_page_config(
    page_title="GSC Keyword Cannibalization Analyzer",
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
    .high-severity {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .medium-severity {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .low-severity {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class AnalysisConfig:
    """Configuration for the cannibalization analysis."""
    min_clicks: int = 1
    click_distribution_threshold: float = 10.0  # Minimum % for cannibalization detection

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
        normalized_columns = {
            col.lower().strip().replace('_', ' '): col 
            for col in csv_columns
        }
        
        for col_type, variations in self.column_variations.items():
            matched_column = None
            
            for variation in variations:
                normalized_variation = variation.lower().strip()
                
                if normalized_variation in normalized_columns:
                    matched_column = normalized_columns[normalized_variation]
                    break
                
                for norm_csv_col, original_csv_col in normalized_columns.items():
                    if (normalized_variation in norm_csv_col or 
                        norm_csv_col in normalized_variation):
                        matched_column = original_csv_col
                        break
                
                if matched_column:
                    break
            
            if matched_column:
                column_mapping[matched_column] = col_type
                logger.debug(f"Mapped: {matched_column} -> {col_type}")
        
        return column_mapping
    
    def process_gsc_data(self, file_path: str) -> pd.DataFrame:
        """Process GSC CSV data with automatic delimiter detection and column mapping."""
        try:
            # Detect delimiter and encoding
            delimiter = self.detect_delimiter(file_path)
            encoding = self.detect_encoding(file_path)
            
            # Load CSV
            df = pd.read_csv(
                file_path,
                sep=delimiter,
                encoding=encoding,
                engine='python',
                on_bad_lines='warn'
            )
            
            logger.info(f"Loaded CSV with shape: {df.shape}")
            
            # Create column mapping
            column_mapping = self.create_column_mapping(df.columns)
            
            # Check required columns
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
            
            # Apply mapping and clean data
            df_mapped = df[list(column_mapping.keys())].rename(columns=column_mapping)
            df_clean = df_mapped.dropna(subset=required_cols)
            
            # Convert numeric columns and remove decimals for display
            numeric_columns = ['clicks', 'impressions', 'ctr', 'position']
            for col in numeric_columns:
                if col in df_clean.columns:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                    # Remove decimals from clicks and impressions as requested
                    if col in ['clicks', 'impressions']:
                        df_clean[col] = df_clean[col].fillna(0).astype(int)
            
            # Remove zero-click rows
            if 'clicks' in df_clean.columns:
                initial_count = len(df_clean)
                df_clean = df_clean[df_clean['clicks'] > 0]
                removed_count = initial_count - len(df_clean)
                if removed_count > 0:
                    logger.info(f"Removed {removed_count} rows with 0 clicks")
            
            # Clean text columns
            for col in ['url', 'query']:
                if col in df_clean.columns:
                    df_clean[col] = df_clean[col].astype(str).str.strip()
                    df_clean = df_clean[df_clean[col] != '']
            
            if len(df_clean) == 0:
                raise ValueError("No valid data remaining after cleaning")
            
            logger.info(f"Data processing complete. Final dataset: {len(df_clean):,} rows")
            return df_clean
            
        except Exception as e:
            error_msg = f"Failed to process GSC data: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

class StreamlitGSCAnalyzer:
    """Main Streamlit application class for GSC Keyword Cannibalization Analysis."""
    
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
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
    
    def render_header(self):
        """Render the main application header."""
        st.markdown('<h1 class="main-header">🔍 GSC Keyword Cannibalization Analyzer</h1>', 
                   unsafe_allow_html=True)
        st.markdown("""
        **Identify TRUE keyword cannibalization issues through click distribution analysis**
        
        This tool analyzes Google Search Console data to identify cases where multiple URLs 
        are actually splitting traffic for the same queries, indicating real cannibalization problems.
        """)
    
    def render_sidebar(self):
        """Render the sidebar with configuration options."""
        st.sidebar.title("⚙️ Configuration")
        
        # File Upload Section
        st.sidebar.subheader("📁 Data Upload")
        self._render_file_upload()
        
        # Analysis Parameters
        st.sidebar.subheader("📊 Analysis Parameters")
        self._render_analysis_parameters()
    
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
    
    def _render_analysis_parameters(self):
        """Render analysis parameters section."""
        config = st.session_state.analysis_config
        
        config.min_clicks = st.sidebar.number_input(
            "Minimum Total Clicks per Query",
            min_value=1,
            max_value=100,
            value=config.min_clicks,
            help="Minimum clicks required across all URLs for a query to be analyzed"
        )
        
        config.click_distribution_threshold = st.sidebar.slider(
            "Click Distribution Threshold (%)",
            min_value=5.0,
            max_value=50.0,
            value=config.click_distribution_threshold,
            step=1.0,
            help="Minimum click percentage for a URL to be considered in cannibalization (10% = at least 2 URLs must have 10%+ of clicks each)"
        )
        
        # Show query count
        if st.session_state.gsc_data is not None:
            df = st.session_state.gsc_data
            eligible_queries = len(df[df['clicks'] >= config.min_clicks])
            st.sidebar.info(f"📊 Queries to analyze: {eligible_queries:,}")
    
    def _process_uploaded_file(self, uploaded_file):
        """Process the uploaded GSC CSV file."""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                temp_path = tmp_file.name
            
            df = self.csv_processor.process_gsc_data(temp_path)
            st.session_state.gsc_data = df
            
            os.unlink(temp_path)
            st.sidebar.success("✅ File processed successfully!")
            
        except Exception as e:
            st.sidebar.error(f"❌ Processing error: {str(e)}")
    
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
            2. **Configure analysis parameters**
            3. **Run the analysis**
            """)
        
        with col2:
            st.markdown("### 📊 What You'll Get")
            st.markdown("""
            - **True Cannibalization Detection** based on click distribution
            - **Severity Classification** (HIGH/MEDIUM/LOW)
            - **URL-based and Query-based Reports**
            - **Number of URLs affected** metrics
            """)
        
        st.markdown("### 📋 Cannibalization Detection Logic")
        st.markdown("""
        **TRUE cannibalization is detected when:**
        - Multiple URLs rank for the same keyword
        - Traffic is **actually being split** between URLs (not dominated by one)
        - At least 2 URLs have significant click distribution (above threshold)
        
        **Severity Levels:**
        - **HIGH**: 5+ URLs affected OR 20-25% distribution with 3-4 URLs
        - **MEDIUM**: 3-4 URLs with moderate distribution
        - **LOW**: 2 URLs with minimal distribution splitting
        
        **NOT cannibalization**: One URL gets 95% of clicks, others get 0-5%
        """)
    
    def _render_analysis_interface(self):
        """Render the main analysis interface."""
        col1, col2 = st.columns([3, 1])
        
        with col1:
            run_analysis = st.button("🚀 Start Cannibalization Analysis", type="primary")
        
        with col2:
            if st.button("🔄 Reset"):
                self._reset_analysis()
        
        if run_analysis:
            self._run_analysis()
        
        if st.session_state.analysis_results is not None:
            self._render_analysis_results()
    
    def _run_analysis(self):
        """Run the cannibalization analysis."""
        df = st.session_state.gsc_data
        config = st.session_state.analysis_config
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("🔄 Analyzing keyword cannibalization...")
            progress_bar.progress(25)
            
            # Analyze cannibalization
            results = self._analyze_cannibalization(df, config)
            progress_bar.progress(75)
            
            # Generate reports
            status_text.text("📊 Generating reports...")
            reports = self._generate_reports(results, df)
            results['reports'] = reports
            progress_bar.progress(100)
            
            st.session_state.analysis_results = results
            status_text.text("✅ Analysis complete!")
            st.success("🎉 Analysis completed successfully!")
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            logger.error(f"Analysis error: {e}")
    
    def _analyze_cannibalization(self, df: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
        """Analyze cannibalization with refined logic based on click distribution."""
        
        # Group by query to analyze click distribution
        query_analysis = []
        cannibalized_queries = []
        urls_affected = set()
        
        query_groups = df.groupby('query')
        
        for query, group in query_groups:
            total_clicks = group['clicks'].sum()
            
            # Only analyze queries with sufficient total clicks
            if total_clicks < config.min_clicks:
                continue
            
            # Calculate click distribution for each URL
            url_distributions = []
            urls_in_query = []
            
            for _, row in group.iterrows():
                url = row['url']
                clicks = row['clicks']
                distribution_pct = (clicks / total_clicks) * 100 if total_clicks > 0 else 0
                
                url_distributions.append({
                    'url': url,
                    'clicks': int(clicks),
                    'distribution_pct': round(distribution_pct, 1)
                })
                urls_in_query.append(url)
            
            # Sort by clicks descending
            url_distributions.sort(key=lambda x: x['clicks'], reverse=True)
            
            # Check for cannibalization: at least 2 URLs with significant distribution
            urls_above_threshold = [u for u in url_distributions 
                                  if u['distribution_pct'] >= config.click_distribution_threshold]
            
            if len(urls_above_threshold) >= 2:
                # This is TRUE cannibalization
                num_urls_affected = len(urls_in_query)
                
                # Determine severity
                max_distribution = max(u['distribution_pct'] for u in urls_above_threshold)
                severity = self._calculate_severity(num_urls_affected, max_distribution, 
                                                 len(urls_above_threshold))
                
                cannibalized_queries.append({
                    'query': query,
                    'total_clicks': int(total_clicks),
                    'num_urls_affected': num_urls_affected,
                    'urls_above_threshold': len(urls_above_threshold),
                    'click_distributions': url_distributions,
                    'severity': severity,
                    'max_distribution_pct': max_distribution
                })
                
                # Add URLs to affected set
                urls_affected.update(urls_in_query)
            
            # Add to general analysis
            query_analysis.append({
                'query': query,
                'total_clicks': int(total_clicks),
                'num_urls': len(urls_in_query),
                'click_distributions': url_distributions,
                'is_cannibalized': len(urls_above_threshold) >= 2
            })
        
        return {
            'query_analysis': query_analysis,
            'cannibalized_queries': cannibalized_queries,
            'total_queries_analyzed': len(query_analysis),
            'total_cannibalized_queries': len(cannibalized_queries),
            'total_urls_affected': len(urls_affected),
            'severity_breakdown': self._calculate_severity_breakdown(cannibalized_queries)
        }
    
    def _calculate_severity(self, num_urls: int, max_distribution: float, urls_above_threshold: int) -> str:
        """Calculate cannibalization severity based on refined criteria."""
        
        # HIGH severity scenarios
        if (num_urls >= 5 and max_distribution >= 10) or \
           (3 <= num_urls <= 4 and 20 <= max_distribution <= 25) or \
           (num_urls >= 2 and max_distribution >= 30):
            return "HIGH"
        
        # MEDIUM severity scenarios  
        elif (3 <= num_urls <= 4 and 15 <= max_distribution < 20) or \
             (num_urls >= 2 and 25 <= max_distribution < 30) or \
             (num_urls >= 5 and max_distribution < 10):
            return "MEDIUM"
        
        # LOW severity (everything else that qualifies as cannibalization)
        else:
            return "LOW"
    
    def _calculate_severity_breakdown(self, cannibalized_queries: List[Dict]) -> Dict[str, int]:
        """Calculate breakdown of cannibalization by severity."""
        breakdown = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for query_data in cannibalized_queries:
            severity = query_data.get('severity', 'LOW')
            breakdown[severity] += 1
        return breakdown
    
    def _generate_reports(self, results: Dict[str, Any], df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Generate comprehensive reports."""
        reports = {}
        
        # URL-Based Report
        reports['url_based'] = self._create_url_based_report(results, df)
        
        # Query-Based Report  
        reports['query_based'] = self._create_query_based_report(results, df)
        
        # Cannibalization Report
        reports['cannibalization'] = self._create_cannibalization_report(results)
        
        return reports
    
    def _create_url_based_report(self, results: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
        """Create URL-based analysis report."""
        # Basic URL statistics
        url_stats = df.groupby('url').agg({
            'query': 'nunique',
            'clicks': 'sum',
            'impressions': 'sum' if 'impressions' in df.columns else lambda x: 0
        }).round(0)
        
        url_stats.columns = ['unique_queries', 'total_clicks', 'total_impressions']
        url_stats = url_stats.astype(int)
        
        # Add cannibalization flags
        cannibalized_urls = set()
        severity_flags = {}
        cannibalized_queries_count = {}
        
        for query_data in results['cannibalized_queries']:
            for url_data in query_data['click_distributions']:
                url = url_data['url']
                if url_data['distribution_pct'] >= st.session_state.analysis_config.click_distribution_threshold:
                    cannibalized_urls.add(url)
                    
                    # Track highest severity for this URL
                    current_severity = severity_flags.get(url, "LOW")
                    new_severity = query_data['severity']
                    
                    if new_severity == "HIGH" or (new_severity == "MEDIUM" and current_severity == "LOW"):
                        severity_flags[url] = new_severity
                    
                    # Count cannibalized queries
                    cannibalized_queries_count[url] = cannibalized_queries_count.get(url, 0) + 1
        
        url_stats['cannibalization_flag'] = url_stats.index.map(lambda x: x in cannibalized_urls)
        url_stats['cannibalization_severity'] = url_stats.index.map(lambda x: severity_flags.get(x, "NONE"))
        url_stats['cannibalized_queries_count'] = url_stats.index.map(lambda x: cannibalized_queries_count.get(x, 0))
        
        # Add top queries
        top_queries = (
            df.groupby('url', group_keys=False)[['query', 'clicks']]
            .apply(lambda g: ', '.join(g.nlargest(3, 'clicks')['query'].values), include_groups=False)
        )
        url_stats['top_queries'] = top_queries
        
        return url_stats.reset_index()
    
    def _create_query_based_report(self, results: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
        """Create query-based cannibalization report."""
        query_report_data = []
        
        for query_data in results['cannibalized_queries']:
            query = query_data['query']
            
            for url_data in query_data['click_distributions']:
                # Get additional metrics from original data
                original_data = df[(df['query'] == query) & (df['url'] == url_data['url'])]
                
                if not original_data.empty:
                    row = original_data.iloc[0]
                    
                    query_report_data.append({
                        'query': query,
                        'url': url_data['url'],
                        'clicks': url_data['clicks'],
                        'click_distribution_pct': url_data['distribution_pct'],
                        'impressions': int(row.get('impressions', 0)) if 'impressions' in row else 0,
                        'ctr': round(row.get('ctr', 0), 2) if 'ctr' in row else 0,
                        'position': round(row.get('position', 0), 1) if 'position' in row else 0,
                        'total_query_clicks': query_data['total_clicks'],
                        'num_competing_urls': query_data['num_urls_affected'],
                        'severity': query_data['severity']
                    })
        
        if query_report_data:
            query_df = pd.DataFrame(query_report_data)
            return query_df.sort_values(['severity', 'total_query_clicks'], 
                                      ascending=[False, False])
        else:
            return pd.DataFrame(columns=['query', 'url', 'clicks', 'click_distribution_pct'])
    
    def _create_cannibalization_report(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Create detailed cannibalization report with actual URL names."""
        cannibalization_data = []
        
        for query_data in results['cannibalized_queries']:
            query = query_data['query']
            distributions = query_data['click_distributions']
            
            # Only include URLs above threshold
            significant_urls = [u for u in distributions 
                              if u['distribution_pct'] >= st.session_state.analysis_config.click_distribution_threshold]
            
            # Create entry for this cannibalized query
            url_list = []
            click_list = []
            distribution_list = []
            
            for i, url_data in enumerate(significant_urls):
                url_list.append(f"URL_{i+1}: {url_data['url'][:50]}...")  # Truncate for display
                click_list.append(url_data['clicks'])
                distribution_list.append(f"{url_data['distribution_pct']}%")
            
            cannibalization_data.append({
                'query': query,
                'severity': query_data['severity'],
                'total_clicks': query_data['total_clicks'],
                'num_urls_competing': len(significant_urls),
                'competing_urls': ' | '.join(url_list),
                'click_distribution': ' | '.join([f"{c} ({d})" for c, d in zip(click_list, distribution_list)]),
                'max_distribution_pct': query_data['max_distribution_pct']
            })
        
        if cannibalization_data:
            cannibalization_df = pd.DataFrame(cannibalization_data)
            return cannibalization_df.sort_values(['severity', 'total_clicks'], 
                                                ascending=[False, False])
        else:
            return pd.DataFrame(columns=['query', 'severity', 'total_clicks', 'num_urls_competing'])
    
    def _render_analysis_results(self):
        """Render analysis results with updated tabs."""
        results = st.session_state.analysis_results
        
        # Create tabs (removed SERP overlap as requested)
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
        """Render analysis overview with corrected metrics."""
        st.markdown("## 📊 Analysis Overview")
        
        # Key metrics with corrected logic
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Queries Analyzed", f"{results['total_queries_analyzed']:,}")
        
        with col2:
            st.metric("TRUE Cannibalization Cases", f"{results['total_cannibalized_queries']:,}")
        
        with col3:
            st.metric("URLs Affected", f"{results['total_urls_affected']:,}")
        
        with col4:
            severity_breakdown = results['severity_breakdown']
            high_severity = severity_breakdown.get('HIGH', 0)
            st.metric("High Severity Cases", f"{high_severity:,}")
        
        # Severity breakdown
        st.markdown("### 🎯 Severity Breakdown")
        severity_breakdown = results['severity_breakdown']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="high-severity">', unsafe_allow_html=True)
            st.metric("🔴 HIGH Severity", f"{severity_breakdown.get('HIGH', 0):,}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="medium-severity">', unsafe_allow_html=True)
            st.metric("🟡 MEDIUM Severity", f"{severity_breakdown.get('MEDIUM', 0):,}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="low-severity">', unsafe_allow_html=True)
            st.metric("🟣 LOW Severity", f"{severity_breakdown.get('LOW', 0):,}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Key insights
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
            st.dataframe(url_report, use_container_width=True, height=400)
            
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
        """Render Query-based report (now properly implemented)."""
        st.markdown("## 🔍 Query-Based Report")
        st.markdown("**Shows each cannibalized query with detailed metrics for every competing URL.**")
        
        reports = results.get('reports', {})
        query_report = reports.get('query_based')
        
        if query_report is not None and not query_report.empty:
            # Filter controls
            col1, col2 = st.columns(2)
            
            with col1:
                severity_filter = st.selectbox(
                    "Filter by Severity",
                    ["All", "HIGH", "MEDIUM", "LOW"],
                    index=0
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
            filtered_report = query_report
            if severity_filter != "All":
                filtered_report = filtered_report[filtered_report['severity'] == severity_filter]
            
            filtered_report = filtered_report.head(max_rows)
            
            st.dataframe(filtered_report, use_container_width=True, height=400)
            
            csv = filtered_report.to_csv(index=False)
            st.download_button(
                label="📥 Download Query Report",
                data=csv,
                file_name=f"query_based_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.success("🎉 No query-level cannibalization detected!")
    
    def _render_cannibalization_report(self, results: Dict[str, Any]):
        """Render detailed cannibalization report with actual URLs."""
        st.markdown("## ⚠️ Cannibalization Report")
        st.markdown("**Detailed view of each cannibalized query with competing URLs and click distributions.**")
        
        reports = results.get('reports', {})
        cannibalization_report = reports.get('cannibalization')
        
        if cannibalization_report is not None and not cannibalization_report.empty:
            st.dataframe(cannibalization_report, use_container_width=True, height=400)
            
            csv = cannibalization_report.to_csv(index=False)
            st.download_button(
                label="📥 Download Cannibalization Report",
                data=csv,
                file_name=f"cannibalization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.success("🎉 No cannibalization detected! Your URLs have proper keyword ownership.")
    
    def _render_visualizations(self, results: Dict[str, Any]):
        """Render data visualizations."""
        st.markdown("## 📈 Visualizations")
        
        cannibalized_queries = results.get('cannibalized_queries', [])
        
        if cannibalized_queries:
            # Severity distribution
            severity_counts = results.get('severity_breakdown', {})
            
            fig_severity = px.pie(
                values=list(severity_counts.values()),
                names=list(severity_counts.keys()),
                title="Cannibalization by Severity Level",
                color_discrete_map={'HIGH': '#f44336', 'MEDIUM': '#ff9800', 'LOW': '#9c27b0'}
            )
            st.plotly_chart(fig_severity, use_container_width=True)
            
            # Click distribution histogram
            max_distributions = [q['max_distribution_pct'] for q in cannibalized_queries]
            
            fig_dist = px.histogram(
                x=max_distributions,
                nbins=20,
                title="Distribution of Maximum Click Percentages in Cannibalized Queries",
                labels={'x': 'Maximum Click Distribution %', 'y': 'Number of Queries'}
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Number of competing URLs
            num_urls = [q['num_urls_affected'] for q in cannibalized_queries]
            
            fig_urls = px.histogram(
                x=num_urls,
                nbins=10,
                title="Number of Competing URLs per Cannibalized Query",
                labels={'x': 'Number of Competing URLs', 'y': 'Number of Queries'}
            )
            st.plotly_chart(fig_urls, use_container_width=True)
        else:
            st.info("No cannibalization data available for visualization.")
    
    def _generate_insights(self, results: Dict[str, Any]) -> List[str]:
        """Generate actionable insights."""
        insights = []
        
        total_cannibalized = results['total_cannibalized_queries']
        total_urls_affected = results['total_urls_affected']
        severity_breakdown = results['severity_breakdown']
        
        if total_cannibalized > 0:
            insights.append(f"🚨 Found {total_cannibalized} queries with TRUE cannibalization affecting {total_urls_affected} URLs")
            
            high_severity = severity_breakdown.get('HIGH', 0)
            if high_severity > 0:
                insights.append(f"⚠️ {high_severity} queries have HIGH severity cannibalization - immediate attention required")
            
            medium_severity = severity_breakdown.get('MEDIUM', 0)
            if medium_severity > 0:
                insights.append(f"📊 {medium_severity} queries have MEDIUM severity - consider consolidation strategies")
            
            # Calculate average URLs affected
            if results.get('cannibalized_queries'):
                avg_urls = sum(q['num_urls_affected'] for q in results['cannibalized_queries']) / len(results['cannibalized_queries'])
                insights.append(f"📈 Average {avg_urls:.1f} URLs competing per cannibalized query")
        else:
            insights.append("✅ No true cannibalization detected - excellent keyword ownership structure!")
        
        return insights
    
    def _reset_analysis(self):
        """Reset analysis results."""
        st.session_state.analysis_results = None
        st.success("🔄 Analysis reset. You can run a new analysis.")
    
    def run(self):
        """Run the main application."""
        try:
            self.render_header()
            self.render_sidebar()
            self.render_main_content()
        except Exception as e:
            st.error(f"Application error: {str(e)}")
            logger.error(f"Application error: {e}")

def main():
    """Main application entry point."""
    try:
        app = StreamlitGSCAnalyzer()
        app.run()
    except Exception as e:
        st.error(f"Failed to initialize application: {str(e)}")
        logger.error(f"Initialization error: {e}")

if __name__ == "__main__":
    main()
