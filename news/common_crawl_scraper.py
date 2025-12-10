"""
Common Crawl Financial News Scraper

Queries Common Crawl dataset to find and download article content from financial news sites.
"""

import requests
import gzip
import json
import io
from typing import List, Dict, Optional, Set
from urllib.parse import urlparse, quote
from bs4 import BeautifulSoup
import time
from dataclasses import dataclass
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optional: CDX Toolkit for fallback
try:
    import cdx_toolkit
    CDX_TOOLKIT_AVAILABLE = True
except ImportError:
    CDX_TOOLKIT_AVAILABLE = False
    logger.warning("cdx_toolkit not installed. Install with: pip install cdx-toolkit")


@dataclass
class Article:
    """Represents a scraped article"""
    url: str
    title: str
    content: str
    domain: str
    crawl_date: str
    fetch_date: str = None

    def to_dict(self) -> Dict:
        return {
            'url': self.url,
            'title': self.title,
            'content': self.content,
            'domain': self.domain,
            'crawl_date': self.crawl_date,
            'fetch_date': self.fetch_date or datetime.now().isoformat()
        }


class CommonCrawlScraper:
    """Scraper for Common Crawl financial news articles"""

    # Financial news domains to scrape
    FINANCIAL_DOMAINS = [
        'bloomberg.com',
        'reuters.com',
        'wsj.com',
        'ft.com',
        'economist.com',
        'forbes.com',
        'marketwatch.com',
        'investing.com',
        'finance.yahoo.com',
        'cnbc.com',
        'investopedia.com',
        'nasdaq.com',
    ]

    CC_INDEX_SERVER = "https://index.commoncrawl.org/"

    def __init__(self, max_articles_per_domain: int = 100, use_cdx_toolkit: bool = False):
        """
        Initialize scraper

        Args:
            max_articles_per_domain: Maximum articles to fetch per domain
            use_cdx_toolkit: Force use of CDX Toolkit instead of direct API
        """
        self.max_articles_per_domain = max_articles_per_domain
        self.use_cdx_toolkit = use_cdx_toolkit
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; FinancialNewsBot/1.0; Research)'
        })
        self.rate_limit_detected = False  # Flag to track if we've hit rate limits

        # Initialize CDX Toolkit client if available
        self.cdx = None
        if CDX_TOOLKIT_AVAILABLE:
            try:
                # Initialize CDX Toolkit with best_effort mode
                # best_effort=True reduces aggressive retries to prevent rate limiting
                self.cdx = cdx_toolkit.CDXFetcher(source='cc', best_effort=True)

                # Try to configure retry behavior via the internal requests module
                try:
                    if hasattr(self.cdx, 'session'):
                        # Some versions expose a session object we can configure
                        from requests.adapters import HTTPAdapter
                        from requests.packages.urllib3.util.retry import Retry

                        # Very conservative retry strategy to avoid rate limits
                        retry_strategy = Retry(
                            total=2,  # Max 2 retries (reduced from 3)
                            backoff_factor=2,  # 2 second backoff (increased from 1)
                            status_forcelist=[429, 500, 502, 503, 504],
                            raise_on_status=False,  # Don't raise on retry-able status codes
                        )
                        adapter = HTTPAdapter(max_retries=retry_strategy)
                        self.cdx.session.mount("http://", adapter)
                        self.cdx.session.mount("https://", adapter)
                        logger.info("CDX Toolkit initialized with conservative retry limits")
                    else:
                        logger.info("CDX Toolkit initialized with best_effort mode")
                except Exception as retry_config_error:
                    logger.debug(f"Could not configure retry behavior: {retry_config_error}")
                    logger.info("CDX Toolkit initialized with best_effort mode")

            except Exception as e:
                logger.warning(f"Failed to initialize CDX Toolkit: {e}")

    def _is_rate_limited(self, error: Exception) -> bool:
        """
        Check if an error indicates rate limiting

        Args:
            error: Exception to check

        Returns:
            True if error indicates rate limiting
        """
        error_msg = str(error).lower()
        rate_limit_indicators = [
            '503', '429',  # HTTP status codes
            'rate limit', 'slow down', 'too many requests',
            'connection aborted', 'remote end closed',
            'remotedisconnected'
        ]
        return any(indicator in error_msg for indicator in rate_limit_indicators)

    def get_available_indexes(self) -> List[str]:
        """
        Get list of available Common Crawl indexes

        Returns:
            List of index names (e.g., ['CC-MAIN-2024-10', ...])
        """
        try:
            response = self.session.get(f"{self.CC_INDEX_SERVER}collinfo.json", timeout=10)
            response.raise_for_status()
            indexes = response.json()
            # Return sorted list of index IDs (most recent first)
            return sorted([idx['id'] for idx in indexes], reverse=True)
        except requests.exceptions.ConnectionError:
            logger.warning("Direct API connection failed. Returning known recent indexes.")
            # Fallback: return known recent indexes
            # These are Common Crawl indexes from 2024-2025 (most recent first)
            # Using only recent indexes as older ones may be deprecated or throttled
            return [
                'CC-MAIN-2024-51',  # December 2024
                'CC-MAIN-2024-46',  # November 2024
                'CC-MAIN-2024-42',  # October 2024
                'CC-MAIN-2024-38',  # September 2024
                'CC-MAIN-2024-33',  # August 2024
            ]
        except Exception as e:
            logger.error(f"Error fetching index list: {e}")
            return []

    def search_domain_cdx_toolkit(self, domain: str, index_name: str, limit: int = 50) -> List[Dict]:
        """
        Search using CDX Toolkit (fallback method)

        Args:
            domain: Domain to search
            index_name: Common Crawl index name
            limit: Maximum results (default 50, reduced from 100 to avoid rate limits)

        Returns:
            List of index records
        """
        if not self.cdx:
            logger.error("CDX Toolkit not available")
            return []

        results = []
        try:
            logger.info(f"Querying {domain} using CDX Toolkit (limit: {limit})")

            # CDX Toolkit query - try only the primary pattern to minimize API calls
            url_patterns = [
                f'{domain}/*',
                # Removed wildcard subdomain pattern to reduce API load
            ]

            count = 0
            attempts = 0
            max_attempts = 1  # Reduced from 2 - only try primary pattern

            for url_pattern in url_patterns:
                if count >= limit or attempts >= max_attempts:
                    break

                attempts += 1
                try:
                    # Query with specific parameters to limit scope
                    # Limit to recent data and successful responses only
                    logger.info(f"Trying pattern: {url_pattern}")

                    # Use iterator with tight limit to prevent deep pagination
                    iter_count = 0
                    max_iter = limit  # Tightened: stop after requested results (was limit * 2)

                    for record in self.cdx.iter(
                        url_pattern,
                        limit=limit - count,
                        filter=['status:200']  # Only successful responses
                    ):
                        iter_count += 1
                        if count >= limit or iter_count > max_iter:
                            logger.info(f"Stopping iteration (count={count}, iter={iter_count})")
                            break

                        # Filter for successful responses
                        status = record.get('status', '')
                        if status != '200':
                            continue

                        # Convert CDX Toolkit record to our format
                        results.append({
                            'url': record.get('url'),
                            'timestamp': record.get('timestamp'),
                            'status': status,
                            'filename': record.get('filename'),
                            'offset': int(record.get('offset', 0)),
                            'length': int(record.get('length', 0)),
                        })
                        count += 1

                        # Rate limiting - sleep every few records to avoid overwhelming the API
                        # Common Crawl CDX API is heavily rate-limited, so we need to be polite
                        if count % 5 == 0:
                            time.sleep(3)  # 3 seconds every 5 records

                    # If we found enough results, no need to try more patterns
                    if count >= limit * 0.8:  # 80% of target
                        logger.info(f"Found sufficient results ({count}), stopping pattern search")
                        break

                    # Sleep between pattern attempts to avoid rate limiting
                    time.sleep(2)

                except KeyboardInterrupt:
                    logger.warning("Search interrupted by user")
                    raise
                except Exception as pattern_error:
                    logger.warning(f"Pattern {url_pattern} failed: {pattern_error}")
                    # Don't try more patterns if we're getting connection errors or rate limited
                    if self._is_rate_limited(pattern_error):
                        logger.error("⚠️  RATE LIMIT detected in CDX Toolkit!")
                        logger.error("Your IP may be temporarily blocked (24h)")
                        logger.error("Stopping all CDX queries to avoid further blocking")
                        self.rate_limit_detected = True
                        break
                    continue

            logger.info(f"Found {len(results)} records for {domain} using CDX Toolkit")
            return results

        except KeyboardInterrupt:
            logger.warning("CDX Toolkit search interrupted by user")
            raise
        except Exception as e:
            logger.error(f"Error searching {domain} with CDX Toolkit: {e}")
            return []

    def search_domain(self, domain: str, index_name: str, limit: int = 50) -> List[Dict]:
        """
        Search for URLs from a specific domain in a Common Crawl index

        Args:
            domain: Domain to search (e.g., 'reuters.com')
            index_name: Common Crawl index name (e.g., 'CC-MAIN-2024-10')
            limit: Maximum number of results to return (default 50, reduced from 100)

        Returns:
            List of index records containing URL, WARC info, etc.
        """
        # Try CDX Toolkit first if forced or if direct API is known to be down
        if self.use_cdx_toolkit and self.cdx:
            return self.search_domain_cdx_toolkit(domain, index_name, limit)

        results = []

        # Common Crawl CDX API query
        # Search for domain/*.* to get all pages
        query_url = f"{self.CC_INDEX_SERVER}{index_name}-index"
        params = {
            'url': f'{domain}/*',
            'output': 'json',
            'limit': limit,
            'filter': 'status:200'  # Only successful fetches
        }

        try:
            logger.info(f"Querying {domain} in index {index_name} (direct API)")
            # Shorter timeout to fail fast and move to CDX Toolkit
            response = self.session.get(query_url, params=params, timeout=15)

            # Check for rate limiting status codes
            if response.status_code in [429, 503]:
                logger.warning(f"Rate limit detected (HTTP {response.status_code}) for {domain}")
                self.rate_limit_detected = True
                raise requests.exceptions.HTTPError(f"Rate limited: {response.status_code}")

            response.raise_for_status()

            # Response is newline-delimited JSON
            for line in response.text.strip().split('\n'):
                if line:
                    try:
                        record = json.loads(line)
                        results.append(record)
                        # Limit results to prevent excessive memory use
                        if len(results) >= limit:
                            break
                    except json.JSONDecodeError:
                        continue

            logger.info(f"Found {len(results)} records for {domain} (direct API)")
            return results

        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.HTTPError) as e:

            # Check if this is a rate limit error
            if self._is_rate_limited(e):
                logger.error(f"⚠️  RATE LIMIT detected for {domain}!")
                logger.error("Common Crawl may have temporarily blocked your IP (24 hour block)")
                logger.error("Recommendation: Wait before continuing or use alternative methods")
                self.rate_limit_detected = True
                return []  # Don't try CDX Toolkit if we're rate limited

            logger.warning(f"Direct API failed for {domain} ({type(e).__name__}): {str(e)[:100]}")

            # Fallback to CDX Toolkit only if not rate limited
            if self.cdx and not self.rate_limit_detected:
                logger.info(f"Falling back to CDX Toolkit for {domain}...")
                time.sleep(2)  # Brief pause before trying CDX Toolkit
                return self.search_domain_cdx_toolkit(domain, index_name, limit)
            else:
                if self.rate_limit_detected:
                    logger.error("Skipping CDX Toolkit fallback due to rate limiting")
                else:
                    logger.error("CDX Toolkit not available for fallback")
                return []

        except KeyboardInterrupt:
            logger.warning("Search interrupted by user")
            raise
        except Exception as e:
            logger.error(f"Unexpected error searching {domain}: {e}")
            # Try CDX Toolkit as last resort
            if self.cdx:
                logger.info("Attempting CDX Toolkit as fallback...")
                return self.search_domain_cdx_toolkit(domain, index_name, limit)
            return []

    def fetch_warc_record(self, warc_filename: str, offset: int, length: int) -> Optional[bytes]:
        """
        Fetch a specific WARC record from Common Crawl S3

        Args:
            warc_filename: WARC file path (e.g., 'crawl-data/...')
            offset: Byte offset in WARC file
            length: Number of bytes to fetch

        Returns:
            Raw WARC record bytes
        """
        # Common Crawl S3 bucket base URL
        s3_base = "https://data.commoncrawl.org/"
        warc_url = f"{s3_base}{warc_filename}"

        # Use HTTP range request to fetch only the specific record
        headers = {'Range': f'bytes={offset}-{offset+length-1}'}

        try:
            response = self.session.get(warc_url, headers=headers, timeout=30)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"Error fetching WARC record: {e}")
            return None

    def parse_warc_record(self, warc_bytes: bytes) -> Optional[str]:
        """
        Parse WARC record to extract HTML content

        Args:
            warc_bytes: Raw WARC record bytes

        Returns:
            HTML content as string
        """
        try:
            # WARC records can be gzipped
            try:
                warc_bytes = gzip.decompress(warc_bytes)
            except:
                pass  # Not gzipped

            warc_content = warc_bytes.decode('utf-8', errors='ignore')

            # WARC format: headers, then HTTP response headers, then content
            # Split on double CRLF to separate sections
            parts = warc_content.split('\r\n\r\n', 2)
            if len(parts) >= 3:
                # Third part is the actual HTML content
                return parts[2]
            elif len(parts) == 2:
                # Sometimes single separator
                return parts[1]
            else:
                return warc_content

        except Exception as e:
            logger.error(f"Error parsing WARC record: {e}")
            return None

    def extract_article_content(self, html: str, url: str) -> Optional[tuple]:
        """
        Extract title and article content from HTML

        Args:
            html: HTML content
            url: Article URL (for domain-specific parsing)

        Returns:
            Tuple of (title, content) or None
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')

            # Remove script, style, and other non-content tags
            for tag in soup(['script', 'style', 'nav', 'header', 'footer',
                            'iframe', 'noscript', 'aside']):
                tag.decompose()

            # Extract title
            title = None
            # Try meta og:title first
            og_title = soup.find('meta', property='og:title')
            if og_title and og_title.get('content'):
                title = og_title['content']
            # Try regular title tag
            elif soup.title:
                title = soup.title.string
            # Try h1
            elif soup.h1:
                title = soup.h1.get_text()

            if not title:
                return None

            title = title.strip()

            # Extract article content
            content_text = ""

            # Try common article containers
            article_selectors = [
                'article',
                '[role="article"]',
                '.article-body',
                '.article-content',
                '.story-body',
                '#article-body',
                '.post-content',
                'main'
            ]

            article_elem = None
            for selector in article_selectors:
                article_elem = soup.select_one(selector)
                if article_elem:
                    break

            if article_elem:
                # Get all paragraphs
                paragraphs = article_elem.find_all('p')
                content_text = '\n\n'.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
            else:
                # Fallback: get all paragraphs
                paragraphs = soup.find_all('p')
                content_text = '\n\n'.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])

            # Basic content validation
            if len(content_text) < 200:  # Too short to be a real article
                return None

            return (title, content_text)

        except Exception as e:
            logger.error(f"Error extracting article content: {e}")
            return None

    def scrape_domain(self, domain: str, index_name: str) -> List[Article]:
        """
        Scrape articles from a specific domain

        Args:
            domain: Domain to scrape
            index_name: Common Crawl index name

        Returns:
            List of Article objects
        """
        articles = []

        # Search for URLs - reduced multiplier to minimize API load
        # Only fetch 1.5x target since not all records will have good content
        records = self.search_domain(domain, index_name, limit=int(self.max_articles_per_domain * 1.5))

        if not records:
            logger.warning(f"No records found for {domain}")
            return articles

        # Process each record
        for record in records[:self.max_articles_per_domain]:
            try:
                url = record.get('url')
                filename = record.get('filename')
                offset = int(record.get('offset', 0))
                length = int(record.get('length', 0))
                timestamp = record.get('timestamp', '')

                if not all([url, filename, offset, length]):
                    continue

                logger.info(f"Fetching: {url}")

                # Fetch WARC record
                warc_bytes = self.fetch_warc_record(filename, offset, length)
                if not warc_bytes:
                    continue

                # Parse WARC to get HTML
                html = self.parse_warc_record(warc_bytes)
                if not html:
                    continue

                # Extract article content
                result = self.extract_article_content(html, url)
                if not result:
                    continue

                title, content = result

                # Create Article object
                article = Article(
                    url=url,
                    title=title,
                    content=content,
                    domain=domain,
                    crawl_date=timestamp
                )

                articles.append(article)
                logger.info(f"Successfully extracted: {title[:50]}...")

                # Rate limiting
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Error processing record: {e}")
                continue

        logger.info(f"Scraped {len(articles)} articles from {domain}")
        return articles

    def scrape_all_domains(self, index_name: Optional[str] = None,
                          domains: Optional[List[str]] = None) -> List[Article]:
        """
        Scrape articles from all configured financial news domains

        Args:
            index_name: Specific index to use (default: most recent)
            domains: List of domains to scrape (default: all financial domains)

        Returns:
            List of all scraped Article objects
        """
        # Get index
        if not index_name:
            indexes = self.get_available_indexes()
            if not indexes:
                logger.error("No indexes available")
                return []
            index_name = indexes[0]  # Use most recent
            logger.info(f"Using index: {index_name}")

        # Get domains
        if not domains:
            domains = self.FINANCIAL_DOMAINS

        all_articles = []

        for domain in domains:
            # Stop if we've been rate limited
            if self.rate_limit_detected:
                logger.error(f"⚠️  Stopping scrape due to rate limiting. Skipping remaining domains.")
                logger.error(f"Successfully scraped {len(all_articles)} articles before rate limit.")
                break

            logger.info(f"\n{'='*60}")
            logger.info(f"Scraping {domain}")
            logger.info(f"{'='*60}")

            articles = self.scrape_domain(domain, index_name)
            all_articles.extend(articles)

            # Rate limiting between domains - be very polite to avoid IP blocks
            logger.info("Sleeping 5 seconds before next domain to avoid rate limiting...")
            time.sleep(5)

        logger.info(f"\n{'='*60}")
        logger.info(f"Total articles scraped: {len(all_articles)}")
        logger.info(f"{'='*60}")

        return all_articles


def save_articles_to_json(articles: List[Article], output_file: str):
    """Save articles to JSON file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump([article.to_dict() for article in articles], f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(articles)} articles to {output_file}")


def save_articles_to_csv(articles: List[Article], output_file: str):
    """Save articles to CSV file"""
    import csv

    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        if articles:
            writer = csv.DictWriter(f, fieldnames=articles[0].to_dict().keys())
            writer.writeheader()
            for article in articles:
                writer.writerow(article.to_dict())
    logger.info(f"Saved {len(articles)} articles to {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Scrape financial news from Common Crawl')
    parser.add_argument('--index', type=str, help='Common Crawl index name (default: most recent)')
    parser.add_argument('--domains', type=str, nargs='+', help='Specific domains to scrape')
    parser.add_argument('--max-per-domain', type=int, default=25,
                       help='Maximum articles per domain (default: 25, reduced to minimize rate limiting)')
    parser.add_argument('--output', type=str, default='financial_news.json',
                       help='Output file path (default: financial_news.json)')
    parser.add_argument('--format', choices=['json', 'csv'], default='json',
                       help='Output format (default: json)')

    args = parser.parse_args()

    # Create scraper
    scraper = CommonCrawlScraper(max_articles_per_domain=args.max_per_domain)

    # Show available indexes
    logger.info("Fetching available indexes...")
    indexes = scraper.get_available_indexes()
    if indexes:
        logger.info(f"Available indexes (showing first 5): {indexes[:5]}")

    # Scrape articles
    articles = scraper.scrape_all_domains(
        index_name=args.index,
        domains=args.domains
    )

    # Save results
    if articles:
        if args.format == 'json':
            save_articles_to_json(articles, args.output)
        else:
            save_articles_to_csv(articles, args.output)
    else:
        logger.warning("No articles scraped")
