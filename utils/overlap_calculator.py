
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from itertools import combinations
import streamlit as st

class OverlapCalculator:
    """Calculate overlaps between queries and SERP results"""

    def calculate_query_overlap_matrix(self, gsc_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Calculate query overlap percentages between all URL pairs

        Args:
            gsc_data: GSC performance data

        Returns:
            DataFrame with URL pairs and their query overlap statistics
        """
        try:
            # Group queries by URL
            url_queries = gsc_data.groupby('url')['query'].apply(set).to_dict()

            # Calculate overlaps between all URL pairs
            overlap_results = []
            url_pairs = list(combinations(url_queries.keys(), 2))

            for url_a, url_b in url_pairs:
                queries_a = url_queries[url_a]
                queries_b = url_queries[url_b]

                # Calculate overlap metrics
                intersection = queries_a.intersection(queries_b)
                union = queries_a.union(queries_b)

                # Calculate overlap percentages
                overlap_count = len(intersection)
                overlap_pct_a = (overlap_count / len(queries_a)) * 100 if queries_a else 0
                overlap_pct_b = (overlap_count / len(queries_b)) * 100 if queries_b else 0

                # Jaccard similarity
                jaccard_similarity = (len(intersection) / len(union)) * 100 if union else 0

                overlap_results.append({
                    'url_a': url_a,
                    'url_b': url_b,
                    'queries_a_count': len(queries_a),
                    'queries_b_count': len(queries_b),
                    'overlap_count': overlap_count,
                    'overlap_percentage_a': round(overlap_pct_a, 2),
                    'overlap_percentage_b': round(overlap_pct_b, 2),
                    'jaccard_similarity': round(jaccard_similarity, 2),
                    'common_queries': ', '.join(list(intersection)[:5])  # First 5 for display
                })

            return pd.DataFrame(overlap_results)

        except Exception as e:
            st.error(f"Error calculating query overlaps: {str(e)}")
            return None

    def calculate_serp_overlaps(self, serp_results: Dict[str, List[str]], gsc_data: pd.DataFrame) -> Dict:
        """
        Calculate SERP overlaps between query pairs

        Args:
            serp_results: Dictionary mapping queries to their SERP URLs
            gsc_data: GSC data to get query pairs that should be compared

        Returns:
            Dictionary with overlap information for query pairs
        """
        try:
            overlaps = {}

            # Get all query combinations from SERP results
            queries = list(serp_results.keys())
            query_pairs = list(combinations(queries, 2))

            for query1, query2 in query_pairs:
                urls1 = set(serp_results[query1])
                urls2 = set(serp_results[query2])

                # Calculate overlap
                intersection = urls1.intersection(urls2)
                union = urls1.union(urls2)

                # Calculate metrics
                overlap_count = len(intersection)
                jaccard_similarity = (overlap_count / len(union)) * 100 if union else 0

                # Store detailed overlap information
                overlaps[(query1, query2)] = {
                    'query1_urls': list(urls1),
                    'query2_urls': list(urls2),
                    'common_urls': list(intersection),
                    'overlap_count': overlap_count,
                    'overlap_percentage': round(jaccard_similarity, 2),
                    'total_unique_urls': len(union)
                }

            return overlaps

        except Exception as e:
            st.error(f"Error calculating SERP overlaps: {str(e)}")
            return {}

    def identify_cannibalization_candidates(self, gsc_data: pd.DataFrame, 
                                          serp_overlaps: Dict = None, 
                                          min_clicks: int = 1,
                                          overlap_threshold: int = 50) -> pd.DataFrame:
        """
        Identify potential keyword cannibalization issues

        Args:
            gsc_data: GSC performance data
            serp_overlaps: SERP overlap data (optional)
            min_clicks: Minimum clicks to consider
            overlap_threshold: Minimum overlap percentage to flag

        Returns:
            DataFrame with cannibalization candidates
        """
        try:
            # Find queries with multiple URLs getting clicks
            multi_url_queries = gsc_data[gsc_data['clicks'] >= min_clicks].groupby('query').agg({
                'url': 'nunique',
                'clicks': 'sum'
            }).reset_index()

            # Filter queries with multiple URLs
            cannibalization_candidates = multi_url_queries[multi_url_queries['url'] > 1]

            # Get detailed breakdown for each candidate query
            detailed_candidates = []

            for _, row in cannibalization_candidates.iterrows():
                query = row['query']
                query_data = gsc_data[
                    (gsc_data['query'] == query) & 
                    (gsc_data['clicks'] >= min_clicks)
                ].sort_values('clicks', ascending=False)

                # Calculate click distribution
                total_clicks = query_data['clicks'].sum()

                for _, url_row in query_data.iterrows():
                    click_share = (url_row['clicks'] / total_clicks) * 100

                    detailed_candidates.append({
                        'query': query,
                        'url': url_row['url'],
                        'clicks': url_row['clicks'],
                        'impressions': url_row['impressions'],
                        'ctr': url_row['ctr'],
                        'position': url_row['position'],
                        'click_share_pct': round(click_share, 2),
                        'total_query_clicks': total_clicks,
                        'competing_urls': len(query_data)
                    })

            return pd.DataFrame(detailed_candidates)

        except Exception as e:
            st.error(f"Error identifying cannibalization candidates: {str(e)}")
            return pd.DataFrame()

    def calculate_jaccard_similarity(self, set1: Set, set2: Set) -> float:
        """Calculate Jaccard similarity between two sets"""
        if not set1 and not set2:
            return 100.0  # Both empty sets are identical

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        return (intersection / union) * 100 if union > 0 else 0.0

    def get_overlap_summary_stats(self, overlap_data: pd.DataFrame) -> Dict:
        """Get summary statistics for overlap analysis"""
        if overlap_data.empty:
            return {}

        stats = {
            'total_url_pairs': len(overlap_data),
            'high_overlap_pairs': len(overlap_data[overlap_data['jaccard_similarity'] >= 50]),
            'medium_overlap_pairs': len(overlap_data[
                (overlap_data['jaccard_similarity'] >= 20) & 
                (overlap_data['jaccard_similarity'] < 50)
            ]),
            'low_overlap_pairs': len(overlap_data[overlap_data['jaccard_similarity'] < 20]),
            'avg_jaccard_similarity': overlap_data['jaccard_similarity'].mean(),
            'max_jaccard_similarity': overlap_data['jaccard_similarity'].max(),
            'avg_overlap_count': overlap_data['overlap_count'].mean()
        }

        return {k: round(v, 2) if isinstance(v, float) else v for k, v in stats.items()}
