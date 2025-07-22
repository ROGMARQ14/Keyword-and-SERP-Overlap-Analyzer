
import pandas as pd
import streamlit as st
from typing import Optional
import logging

class GSCProcessor:
    """Process and validate Google Search Console data"""

    def __init__(self):
        self.required_columns = ['query', 'url', 'clicks', 'impressions', 'ctr', 'position']

    def load_and_validate_data(self, uploaded_file) -> Optional[pd.DataFrame]:
        """Load and validate GSC CSV data"""
        try:
            # Read CSV file
            df = pd.read_csv(uploaded_file)

            # Standardize column names (handle common variations)
            column_mapping = {
                'Query': 'query',
                'Queries': 'query', 
                'Top queries': 'query',
                'URL': 'url',
                'URLs': 'url',
                'Top pages': 'url',
                'Page': 'url',
                'Clicks': 'clicks',
                'Impressions': 'impressions',
                'CTR': 'ctr',
                'Click-through rate': 'ctr',
                'Position': 'position',
                'Average position': 'position',
                'Avg. position': 'position'
            }

            # Apply column mapping
            df = df.rename(columns=column_mapping)

            # Check for required columns
            missing_columns = set(self.required_columns) - set(df.columns)
            if missing_columns:
                st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
                return None

            # Data cleaning and validation
            df = self.clean_data(df)

            return df

        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            return None

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare GSC data"""
        # Remove rows with missing essential data
        df = df.dropna(subset=['query', 'url'])

        # Clean numeric columns
        for col in ['clicks', 'impressions', 'position']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Handle CTR column (might be percentage or decimal)
        if df['ctr'].dtype == 'object':
            # Remove percentage sign if present
            df['ctr'] = df['ctr'].str.replace('%', '').astype(float) / 100

        # Ensure CTR is in decimal format (0-1)
        if df['ctr'].max() > 1:
            df['ctr'] = df['ctr'] / 100

        # Remove invalid rows
        df = df[df['clicks'] >= 0]
        df = df[df['impressions'] >= 0]
        df = df[df['position'] > 0]

        # Clean URLs (remove fragments and parameters for grouping)
        df['clean_url'] = df['url'].apply(self.clean_url)

        return df

    def clean_url(self, url: str) -> str:
        """Clean URL for better grouping"""
        # Remove URL fragments and some parameters
        if '#' in url:
            url = url.split('#')[0]

        # Remove common tracking parameters
        tracking_params = ['utm_source', 'utm_medium', 'utm_campaign', 'utm_content', 'utm_term', 
                          'gclid', 'fbclid', 'ref', 'source']

        if '?' in url:
            base_url, params = url.split('?', 1)
            if '&' in params:
                param_list = params.split('&')
                filtered_params = [p for p in param_list 
                                 if not any(p.startswith(tp + '=') for tp in tracking_params)]
                if filtered_params:
                    url = base_url + '?' + '&'.join(filtered_params)
                else:
                    url = base_url

        return url

    def get_query_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get query-level statistics"""
        query_stats = df.groupby('query').agg({
            'clicks': 'sum',
            'impressions': 'sum',
            'url': 'nunique',
            'position': 'mean'
        }).round(2)

        query_stats['ctr'] = (query_stats['clicks'] / query_stats['impressions']).round(4)
        query_stats = query_stats.rename(columns={'url': 'unique_urls'})

        return query_stats.sort_values('clicks', ascending=False)

    def get_url_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get URL-level statistics"""
        url_stats = df.groupby('url').agg({
            'clicks': 'sum',
            'impressions': 'sum',
            'query': 'nunique',
            'position': 'mean'
        }).round(2)

        url_stats['ctr'] = (url_stats['clicks'] / url_stats['impressions']).round(4)
        url_stats = url_stats.rename(columns={'query': 'unique_queries'})

        return url_stats.sort_values('clicks', ascending=False)
