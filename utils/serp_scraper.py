
import requests
import time
import streamlit as st
from typing import Dict, List, Optional
import logging

class SERPScraper:
    """Scrape SERP results using Serper API"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://google.serper.dev/search"
        self.headers = {
            'X-API-KEY': api_key,
            'Content-Type': 'application/json'
        }

    def get_serp_results(self, query: str, num_results: int = 10, country: str = "us", language: str = "en") -> Optional[List[str]]:
        """
        Get SERP results for a query using Serper API

        Args:
            query: Search query
            num_results: Number of results to retrieve (max 10 for free tier)
            country: Country code for localization
            language: Language code

        Returns:
            List of URLs from organic search results
        """
        try:
            payload = {
                "q": query,
                "num": min(num_results, 10),  # Limit to 10 for API limits
                "gl": country,
                "hl": language
            }

            response = requests.post(
                self.base_url, 
                headers=self.headers, 
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()

                # Extract organic results URLs
                urls = []
                if 'organic' in data:
                    for result in data['organic'][:num_results]:
                        if 'link' in result:
                            urls.append(result['link'])

                return urls

            elif response.status_code == 429:
                st.warning(f"Rate limit reached. Waiting 60 seconds...")
                time.sleep(60)
                return self.get_serp_results(query, num_results, country, language)

            else:
                st.error(f"API Error {response.status_code}: {response.text}")
                return None

        except requests.exceptions.RequestException as e:
            st.error(f"Request error for query '{query}': {str(e)}")
            return None

        except Exception as e:
            st.error(f"Unexpected error for query '{query}': {str(e)}")
            return None

    def batch_scrape_serps(self, queries: List[str], delay: float = 1.0) -> Dict[str, List[str]]:
        """
        Scrape SERP results for multiple queries with rate limiting

        Args:
            queries: List of search queries
            delay: Delay between requests in seconds

        Returns:
            Dictionary mapping queries to their SERP URLs
        """
        results = {}

        for i, query in enumerate(queries):
            st.write(f"Scraping {i+1}/{len(queries)}: {query}")

            serp_urls = self.get_serp_results(query)
            if serp_urls:
                results[query] = serp_urls

            # Rate limiting
            if delay > 0 and i < len(queries) - 1:
                time.sleep(delay)

        return results

    def validate_api_key(self) -> bool:
        """Test if the API key is valid"""
        try:
            test_query = "test"
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json={"q": test_query, "num": 1},
                timeout=10
            )

            return response.status_code in [200, 429]  # 429 is rate limit, but key is valid

        except:
            return False

    def get_api_usage_info(self) -> Dict:
        """Get information about API usage (if available from headers)"""
        # Note: Serper doesn't typically return usage info in headers
        # This is a placeholder for potential future functionality
        return {
            "status": "unknown",
            "remaining_calls": "unknown",
            "reset_time": "unknown"
        }
