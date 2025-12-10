"""
Test the direct Common Crawl API without CDX Toolkit fallback
"""

from common_crawl_scraper import CommonCrawlScraper
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print('=' * 70)
print('Testing Direct Common Crawl API (No CDX Toolkit Fallback)')
print('=' * 70)
print()

# Initialize scraper WITHOUT CDX Toolkit
logger.info("Initializing scraper (CDX Toolkit disabled)...")
scraper = CommonCrawlScraper(
    max_articles_per_domain=5,
    use_cdx_toolkit=False  # Disable CDX Toolkit
)

# Manually disable CDX Toolkit to prevent fallback
scraper.cdx = None

print(f"✓ Scraper initialized")
print(f"  - CDX Toolkit: Disabled (to avoid hanging)")
print()

# Get available indexes
logger.info("Fetching available indexes...")
indexes = scraper.get_available_indexes()

if indexes:
    print(f"✓ Found {len(indexes)} available indexes")
    print(f"  Most recent: {indexes[0]}")
else:
    print("Using fallback indexes")
    indexes = [
        'CC-MAIN-2024-51',
        'CC-MAIN-2024-46',
        'CC-MAIN-2024-42'
    ]

print()

# Try multiple domains to increase chance of success
test_domains = ['cnbc.com', 'bloomberg.com', 'marketwatch.com']
test_index = indexes[0]

print(f"Testing with index: {test_index}")
print(f"Will try domains: {', '.join(test_domains)}")
print("Using direct API with 15-second timeout...")
print()

results_found = False

for domain in test_domains:
    print(f"Trying {domain}...", end=' ', flush=True)

    try:
        records = scraper.search_domain(domain, test_index, limit=5)

        if records:
            print(f"✓ Found {len(records)} records!")
            results_found = True

            for i, rec in enumerate(records[:2], 1):
                url = rec.get('url', 'N/A')
                print(f"  {i}. {url[:60]}...")

            break  # Stop after first successful domain
        else:
            print("No results")

    except Exception as e:
        print(f"Error: {str(e)[:50]}")

print()

if results_found:
    print('=' * 70)
    print('✓ Direct API is working!')
    print()
    print('Recommendation: Use the direct API approach')
    print('  - Disable CDX Toolkit to avoid hanging')
    print('  - Use recent indexes (2024)')
    print('  - 15-second timeout per request (fails fast)')
    print('=' * 70)
else:
    print('=' * 70)
    print('✗ Direct API returned no results')
    print()
    print('This suggests Common Crawl API is experiencing issues.')
    print()
    print('Alternative approaches:')
    print('  1. Wait for Common Crawl API to recover')
    print('  2. Use alternative data sources (see README)')
    print('  3. Download CC index files directly from S3')
    print('  4. Use CC Columnar Index (Parquet) for large-scale queries')
    print('=' * 70)
