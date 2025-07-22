"""
GSC Performance & SERP Overlap Analyzer - Streamlined Version
Main Streamlit Application for analyzing GSC data and SERP overlaps
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional
from itertools import combinations
import re

# Page Configuration
st.set_page_config(
    page_title="GSC SERP Overlap Analyzer",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)


class GSCProcessor:
    """Simplified GSC data processor with flexible column mapping."""
    
    # Column variations (case-insensitive)
    COLUMN_MAPPING = {
        'query': ['query', 'queries', 'keyword', 'keywords', 'keyqords', 'search term'],
        'url': ['url', 'urls', 'page', 'pages', 'landing page', 'address', 'destination url'],
        'clicks': ['click', 'clicks', 'total clicks'],
        'impressions': ['impression', 'impressions', 'total impressions'],
        'ctr': ['ctr', 'url ctr', 'click through rate', 'clickthrough rate'],
        'position': ['position', 'positions', 'avg pos', 'avg. pos', 'avg position', 'average position']
    }
    
    def process_csv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process uploaded CSV with flexible column mapping."""
        # Create case-insensitive column mapping
        df_columns = {col.lower().strip(): col for col in df.columns}
        
        column_map = {}
        for standard_name, variations in self.COLUMN_MAPPING.items():
            for variation in variations:
                normalized_var = variation.lower().strip()
                if normalized_var in df_columns:
                    column_map[df_columns[normalized_var]] = standard_name
                    break
        
        # Check required columns
        required = ['query', 'url', 'clicks']
        mapped_required = [col for col in required if col in column_map.values()]
        
        if len(mapped_required) < len(required):
            missing = set(required) - set(column_map.values())
            raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")
        
        # Select and rename columns
        selected_cols = list(column_map.keys())
        df_processed = df[selected_cols].rename(columns=column_map)
        
        # Clean data
        df_processed = df_processed.dropna(subset=['query', 'url', 'clicks'])
        df_processed['clicks'] = pd.to_numeric(df_processed['clicks'], errors='coerce')
        df_processed = df_processed[df_processed['clicks'] > 0]  # Only queries with clicks
        
        # Clean text columns
        df_processed['query'] = df_processed['query'].astype(str).str.strip()
        df_processed['url'] = df_processed['url'].astype(str).str.strip()
        
        return df_processed


class SERPAnalyzer:
    """Simplified SERP analysis using Serper API."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://google.serper.dev/search"
    
    def get_serp_results(self, query: str, num_results: int = 10) -> List[str]:
        """Get SERP results for a query."""
        headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }
        
        payload = {
            'q': query,
            'num': num_results
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if 'organic' in data:
                return [result.get('link', '') for result in data['organic']]
            return []
            
        except Exception as e:
            st.warning(f"SERP API error for query '{query}': {str(e)}")
            return []


class OverlapAnalyzer:
    """Analyze query and SERP overlaps."""
    
    def analyze_query_overlaps(self, df: pd.DataFrame) -> List[Dict]:
        """Analyze query overlaps between URLs."""
        url_queries = df.groupby('url')['query'].apply(set).to_dict()
        overlaps = []
        
        for url1, url2 in combinations(url_queries.keys(), 2):
            queries1, queries2 = url_queries[url1], url_queries[url2]
            shared = queries1.intersection(queries2)
            
            if len(shared) > 0:
                overlaps.append({
                    'url1': url1,
                    'url2': url2,
                    'shared_queries': len(shared),
                    'url1_total': len(queries1),
                    'url2_total': len(queries2),
                    'url1_overlap_pct': len(shared) / len(queries1) * 100,
                    'url2_overlap_pct': len(shared) / len(queries2) * 100
                })
        
        return overlaps
    
    def analyze_serp_overlaps(self, queries: List[str], serp_analyzer: SERPAnalyzer, 
                            progress_callback=None) -> List[Dict]:
        """Analyze SERP overlaps between query pairs."""
        serp_data = {}
        overlaps = []
        
        # Get SERP data
        for i, query in enumerate(queries):
            if progress_callback:
                progress_callback(i / len(queries))
            
            serp_results = serp_analyzer.get_serp_results(query)
            if serp_results:
                serp_data[query] = set(serp_results)
            time.sleep(1)  # Rate limiting
        
        # Calculate overlaps
        for query1, query2 in combinations(serp_data.keys(), 2):
            serp1, serp2 = serp_data[query1], serp_data[query2]
            intersection = serp1.intersection(serp2)
            union = serp1.union(serp2)
            
            if len(union) > 0:
                jaccard_sim = len(intersection) / len(union) * 100
                if jaccard_sim >= 10:  # Only meaningful overlaps
                    overlaps.append({
                        'query1': query1,
                        'query2': query2,
                        'serp_overlap_pct': jaccard_sim,
                        'shared_domains': len(intersection)
                    })
        
        return overlaps


# Main Application
def main():
    st.title("🔍 GSC Performance & SERP Overlap Analyzer")
    st.markdown("**Identify keyword cannibalization through query and SERP overlap analysis**")
    
    # Sidebar Configuration
    st.sidebar.header("⚙️ Configuration")
    
    # File Upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload GSC Performance CSV",
        type=['csv'],
        help="Upload your Google Search Console performance data"
    )
    
    # API Configuration
    serp_api_key = st.sidebar.text_input(
        "Serper API Key",
        type="password",
        help="Get your API key from serper.dev"
    )
    
    # Analysis Parameters
    min_clicks = st.sidebar.number_input("Minimum Clicks", min_value=1, value=1)
    max_queries = st.sidebar.number_input("Max Queries for SERP Analysis", min_value=10, max_value=500, value=50)
    serp_threshold = st.sidebar.slider("SERP Overlap Threshold (%)", 0.0, 100.0, 50.0)
    
    # Process uploaded file
    if uploaded_file is not None:
        try:
            # Load and process data
            df_raw = pd.read_csv(uploaded_file)
            processor = GSCProcessor()
            df = processor.process_csv(df_raw)
            
            st.success(f"✅ Data loaded: {len(df):,} rows")
            
            # Data summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total URLs", df['url'].nunique())
            with col2:
                st.metric("Total Queries", df['query'].nunique())
            with col3:
                st.metric("Total Clicks", f"{df['clicks'].sum():,.0f}")
            
            # Analysis Tabs
            tab1, tab2, tab3 = st.tabs(["📊 Query Overlap Analysis", "🔍 SERP Overlap Analysis", "📋 Reports"])
            
            with tab1:
                st.subheader("Query Overlap Analysis")
                if st.button("🚀 Analyze Query Overlaps"):
                    analyzer = OverlapAnalyzer()
                    overlaps = analyzer.analyze_query_overlaps(df)
                    
                    if overlaps:
                        overlap_df = pd.DataFrame(overlaps)
                        st.dataframe(overlap_df, use_container_width=True)
                        
                        # Download option
                        csv = overlap_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Query Overlap Report",
                            csv,
                            f"query_overlaps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            "text/csv"
                        )
                    else:
                        st.info("No significant query overlaps found")
            
            with tab2:
                st.subheader("SERP Overlap Analysis")
                
                if not serp_api_key:
                    st.warning("⚠️ Please enter your Serper API key to perform SERP analysis")
                else:
                    # Filter queries for SERP analysis
                    top_queries = df[df['clicks'] >= min_clicks].nlargest(max_queries, 'clicks')['query'].unique()
                    
                    st.info(f"Will analyze {len(top_queries)} queries with ≥{min_clicks} clicks")
                    
                    if st.button("🚀 Analyze SERP Overlaps"):
                        serp_analyzer = SERPAnalyzer(serp_api_key)
                        analyzer = OverlapAnalyzer()
                        
                        # Progress tracking
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        def update_progress(pct):
                            progress_bar.progress(pct)
                            status_text.text(f"Processing... {pct*100:.1f}%")
                        
                        serp_overlaps = analyzer.analyze_serp_overlaps(
                            top_queries.tolist(), 
                            serp_analyzer, 
                            update_progress
                        )
                        
                        status_text.text("✅ Analysis complete!")
                        
                        if serp_overlaps:
                            serp_df = pd.DataFrame(serp_overlaps)
                            high_overlap = serp_df[serp_df['serp_overlap_pct'] >= serp_threshold]
                            
                            st.subheader(f"High SERP Overlaps (≥{serp_threshold}%)")
                            st.dataframe(high_overlap, use_container_width=True)
                            
                            # Download option
                            csv = serp_df.to_csv(index=False)
                            st.download_button(
                                "📥 Download SERP Overlap Report",
                                csv,
                                f"serp_overlaps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                "text/csv"
                            )
                        else:
                            st.info("No significant SERP overlaps found")
            
            with tab3:
                st.subheader("📋 Combined Reports")
                
                # URL-Based Report
                st.subheader("URL-Based Summary")
                url_summary = df.groupby('url').agg({
                    'clicks': 'sum',
                    'impressions': 'sum',
                    'query': 'nunique'
                }).round(2)
                url_summary.columns = ['Total Clicks', 'Total Impressions', 'Unique Queries']
                
                st.dataframe(url_summary, use_container_width=True)
                
                # Query-Based Report
                st.subheader("Query Performance")
                query_summary = df.groupby('query').agg({
                    'clicks': 'sum',
                    'url': 'nunique'
                }).round(2)
                query_summary.columns = ['Total Clicks', 'URLs Ranking']
                query_summary = query_summary.sort_values('Total Clicks', ascending=False).head(20)
                
                st.dataframe(query_summary, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure your CSV contains columns for Query, URL, and Clicks")
    
    else:
        st.info("👆 Please upload your GSC performance CSV file to begin analysis")


if __name__ == "__main__":
    main()
