
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import streamlit as st
from datetime import datetime

class ReportGenerator:
    """Generate comprehensive reports for GSC and SERP overlap analysis"""

    def generate_url_report(self, gsc_data: pd.DataFrame, serp_overlaps: Optional[Dict] = None, 
                           serp_threshold: int = 50) -> Optional[pd.DataFrame]:
        """
        Generate URL-based report with query overlap and SERP analysis

        Args:
            gsc_data: GSC performance data
            serp_overlaps: SERP overlap data (optional)
            serp_threshold: Minimum SERP overlap percentage to include

        Returns:
            DataFrame with URL-level analysis
        """
        try:
            # Calculate URL-level metrics
            url_metrics = gsc_data.groupby('url').agg({
                'clicks': 'sum',
                'impressions': 'sum',
                'query': ['count', 'nunique'],
                'position': 'mean',
                'ctr': 'mean'
            }).round(3)

            # Flatten column names
            url_metrics.columns = ['total_clicks', 'total_impressions', 'total_queries', 
                                 'unique_queries', 'avg_position', 'avg_ctr']

            # Calculate additional metrics
            url_metrics['calculated_ctr'] = (url_metrics['total_clicks'] / 
                                           url_metrics['total_impressions']).round(4)

            # Add query overlap analysis
            url_metrics['top_queries'] = gsc_data.groupby('url').apply(
                lambda x: ', '.join(x.nlargest(5, 'clicks')['query'].tolist())
            )

            # Add SERP overlap information if available
            if serp_overlaps:
                url_metrics['high_serp_overlaps'] = self._count_high_serp_overlaps(
                    gsc_data, serp_overlaps, serp_threshold
                )

            # Reset index to make URL a column
            url_report = url_metrics.reset_index()

            # Sort by total clicks descending
            url_report = url_report.sort_values('total_clicks', ascending=False)

            return url_report

        except Exception as e:
            st.error(f"Error generating URL report: {str(e)}")
            return None

    def generate_query_report(self, gsc_data: pd.DataFrame, serp_overlaps: Optional[Dict] = None) -> Optional[pd.DataFrame]:
        """
        Generate detailed query-based report with SERP overlap information

        Args:
            gsc_data: GSC performance data
            serp_overlaps: SERP overlap data (optional)

        Returns:
            DataFrame with query-level analysis including SERP overlaps
        """
        try:
            query_report_data = []

            if serp_overlaps:
                # Create report with SERP overlap information
                processed_pairs = set()

                for (query1, query2), overlap_info in serp_overlaps.items():
                    # Avoid duplicate pairs
                    pair_key = tuple(sorted([query1, query2]))
                    if pair_key in processed_pairs:
                        continue
                    processed_pairs.add(pair_key)

                    # Get GSC data for both queries
                    query1_data = gsc_data[gsc_data['query'] == query1]
                    query2_data = gsc_data[gsc_data['query'] == query2]

                    if not query1_data.empty and not query2_data.empty:
                        # Get top URL for each query
                        top_url_q1 = query1_data.nlargest(1, 'clicks').iloc[0]
                        top_url_q2 = query2_data.nlargest(1, 'clicks').iloc[0]

                        # Add rows for both queries
                        query_report_data.append({
                            'url': top_url_q1['url'],
                            'query': query1,
                            'serp_overlap_pct': overlap_info['overlap_percentage'],
                            'overlapping_query': query2,
                            'clicks': top_url_q1['clicks'],
                            'impressions': top_url_q1['impressions'],
                            'ctr': top_url_q1['ctr'],
                            'position': top_url_q1['position'],
                            'common_serp_urls': len(overlap_info['common_urls']),
                            'total_serp_urls': overlap_info['total_unique_urls']
                        })

                        query_report_data.append({
                            'url': top_url_q2['url'],
                            'query': query2,
                            'serp_overlap_pct': overlap_info['overlap_percentage'],
                            'overlapping_query': query1,
                            'clicks': top_url_q2['clicks'],
                            'impressions': top_url_q2['impressions'],
                            'ctr': top_url_q2['ctr'],
                            'position': top_url_q2['position'],
                            'common_serp_urls': len(overlap_info['common_urls']),
                            'total_serp_urls': overlap_info['total_unique_urls']
                        })

                if query_report_data:
                    query_report = pd.DataFrame(query_report_data)
                    # Sort by SERP overlap percentage descending
                    query_report = query_report.sort_values('serp_overlap_pct', ascending=False)

                    # Remove duplicate rows (same URL-query combination)
                    query_report = query_report.drop_duplicates(subset=['url', 'query'])

                    return query_report
                else:
                    st.info("No SERP overlap data available for query report")
                    return None
            else:
                # Generate basic query report without SERP data
                query_metrics = gsc_data.groupby(['url', 'query']).agg({
                    'clicks': 'sum',
                    'impressions': 'sum',
                    'ctr': 'mean',
                    'position': 'mean'
                }).round(3).reset_index()

                return query_metrics.sort_values('clicks', ascending=False)

        except Exception as e:
            st.error(f"Error generating query report: {str(e)}")
            return None

    def generate_cannibalization_report(self, gsc_data: pd.DataFrame, min_clicks: int = 1) -> Optional[pd.DataFrame]:
        """
        Generate a focused report on potential keyword cannibalization issues

        Args:
            gsc_data: GSC performance data
            min_clicks: Minimum clicks to consider for cannibalization

        Returns:
            DataFrame with cannibalization analysis
        """
        try:
            # Find queries with multiple URLs getting significant traffic
            multi_url_queries = gsc_data[gsc_data['clicks'] >= min_clicks].groupby('query').agg({
                'url': 'nunique',
                'clicks': 'sum'
            }).reset_index()

            cannibalization_queries = multi_url_queries[multi_url_queries['url'] > 1]['query'].tolist()

            if not cannibalization_queries:
                return pd.DataFrame()

            # Detailed analysis for each cannibalization query
            cannibalization_data = []

            for query in cannibalization_queries:
                query_data = gsc_data[
                    (gsc_data['query'] == query) & 
                    (gsc_data['clicks'] >= min_clicks)
                ].sort_values('clicks', ascending=False)

                total_clicks = query_data['clicks'].sum()

                for i, (_, row) in enumerate(query_data.iterrows()):
                    cannibalization_data.append({
                        'query': query,
                        'url': row['url'],
                        'url_rank': i + 1,
                        'clicks': row['clicks'],
                        'impressions': row['impressions'],
                        'ctr': row['ctr'],
                        'position': row['position'],
                        'click_share_pct': round((row['clicks'] / total_clicks) * 100, 2),
                        'total_query_clicks': total_clicks,
                        'competing_urls_count': len(query_data),
                        'cannibalization_severity': self._assess_cannibalization_severity(
                            len(query_data), (row['clicks'] / total_clicks) * 100
                        )
                    })

            cannibalization_report = pd.DataFrame(cannibalization_data)
            return cannibalization_report.sort_values(['total_query_clicks', 'clicks'], ascending=[False, False])

        except Exception as e:
            st.error(f"Error generating cannibalization report: {str(e)}")
            return None

    def _count_high_serp_overlaps(self, gsc_data: pd.DataFrame, serp_overlaps: Dict, threshold: int) -> pd.Series:
        """Count high SERP overlaps for each URL"""
        url_overlap_counts = {}

        # Get queries for each URL
        url_queries = gsc_data.groupby('url')['query'].apply(set).to_dict()

        for url, queries in url_queries.items():
            high_overlap_count = 0

            for (query1, query2), overlap_info in serp_overlaps.items():
                if (query1 in queries or query2 in queries) and overlap_info['overlap_percentage'] >= threshold:
                    high_overlap_count += 1

            url_overlap_counts[url] = high_overlap_count

        return pd.Series(url_overlap_counts)

    def _assess_cannibalization_severity(self, competing_urls: int, click_share: float) -> str:
        """Assess the severity of keyword cannibalization"""
        if competing_urls >= 5:
            return "High"
        elif competing_urls >= 3 and click_share < 50:
            return "Medium"
        elif competing_urls >= 2 and click_share < 70:
            return "Low"
        else:
            return "Minimal"

    def generate_summary_report(self, gsc_data: pd.DataFrame, serp_overlaps: Optional[Dict] = None) -> Dict:
        """Generate summary statistics for the analysis"""
        try:
            summary = {
                'total_urls': gsc_data['url'].nunique(),
                'total_queries': gsc_data['query'].nunique(),
                'total_clicks': int(gsc_data['clicks'].sum()),
                'total_impressions': int(gsc_data['impressions'].sum()),
                'avg_ctr': round(gsc_data['ctr'].mean() * 100, 2),
                'avg_position': round(gsc_data['position'].mean(), 2),
                'queries_with_multiple_urls': len(
                    gsc_data.groupby('query').agg({'url': 'nunique'}).query('url > 1')
                ),
                'urls_with_multiple_queries': len(
                    gsc_data.groupby('url').agg({'query': 'nunique'}).query('query > 1')
                )
            }

            if serp_overlaps:
                high_overlap_pairs = sum(1 for overlap in serp_overlaps.values() 
                                       if overlap['overlap_percentage'] >= 50)
                summary['high_serp_overlap_pairs'] = high_overlap_pairs
                summary['total_serp_comparisons'] = len(serp_overlaps)

            return summary

        except Exception as e:
            st.error(f"Error generating summary report: {str(e)}")
            return {}

    def export_all_reports(self, gsc_data: pd.DataFrame, serp_overlaps: Optional[Dict] = None) -> Dict[str, pd.DataFrame]:
        """Export all reports as a dictionary of DataFrames"""
        reports = {}

        url_report = self.generate_url_report(gsc_data, serp_overlaps)
        if url_report is not None:
            reports['url_report'] = url_report

        query_report = self.generate_query_report(gsc_data, serp_overlaps)
        if query_report is not None:
            reports['query_report'] = query_report

        cannibalization_report = self.generate_cannibalization_report(gsc_data)
        if cannibalization_report is not None and not cannibalization_report.empty:
            reports['cannibalization_report'] = cannibalization_report

        return reports
