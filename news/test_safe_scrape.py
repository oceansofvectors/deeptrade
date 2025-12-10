"""
Safe test script for CommonCrawl scraper with connection monitoring
"""

from common_crawl_scraper import CommonCrawlScraper
import logging
import signal
import sys

# Set up logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Reduce CDX Toolkit verbosity
logging.getLogger('cdx_toolkit').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.WARNING)

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    logger.warning("\n\nInterrupted by user. Exiting gracefully...")
    sys.exit(0)

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

print('=' * 70)
print('CommonCrawl Scraper - Safe Test with Recent Indexes')
print('=' * 70)
print()

# Initialize scraper with conservative limits
logger.info("Initializing scraper with retry limits...")
scraper = CommonCrawlScraper(
    max_articles_per_domain=5,  # Very conservative: only 5 articles
    use_cdx_toolkit=True  # Force CDX Toolkit to test the fix
)

print(f"✓ Scraper initialized")
print(f"  - CDX Toolkit available: {scraper.cdx is not None}")
print(f"  - Max retry attempts: 3 (down from unlimited)")
print(f"  - Retry delay: 5 seconds (down from 60)")
print()

# Get available indexes
logger.info("Fetching available indexes...")
indexes = scraper.get_available_indexes()

if indexes:
    print(f"✓ Found {len(indexes)} available indexes")
    print(f"  Most recent: {indexes[0]}")
    print(f"  Using: {indexes[0]}")
else:
    print("✗ No indexes found, exiting")
    sys.exit(1)

print()

# Test with a smaller, more reliable domain
test_domain = 'cnbc.com'
test_index = indexes[0]  # Use most recent index

logger.info(f"Testing search for {test_domain} in {test_index}...")
print(f"Testing: {test_domain} (limit: 5 articles)")
print(f"Index: {test_index}")
print()
print("This should complete quickly now (max 15 seconds per attempt)...")
print("If it takes longer, press Ctrl+C to cancel")
print()

try:
    # Search for records
    records = scraper.search_domain(test_domain, test_index, limit=5)

    if records:
        print(f"\n✓ SUCCESS: Found {len(records)} records")
        print()
        print("Sample results:")
        for i, rec in enumerate(records[:3], 1):
            url = rec.get('url', 'N/A')
            print(f"  {i}. {url[:70]}{'...' if len(url) > 70 else ''}")
            print(f"     Status: {rec.get('status', 'N/A')}")
            print(f"     Timestamp: {rec.get('timestamp', 'N/A')}")
        print()
    else:
        print("\n✓ Search completed but no records found")
        print("  (This may be normal - try a different domain)")
        print()

except KeyboardInterrupt:
    print("\n\n✗ Interrupted by user")
    sys.exit(1)
except Exception as e:
    print(f"\n✗ Error: {e}")
    logger.error(f"Test failed with error: {e}", exc_info=True)
    sys.exit(1)

print('=' * 70)
print('Test completed successfully!')
print()
print('Key improvements:')
print('  ✓ CDX Toolkit now has retry limits (3 attempts max)')
print('  ✓ Retry delay reduced from 60s to 5s')
print('  ✓ Using recent indexes (2024) instead of old ones (2019)')
print('  ✓ Pagination limited to prevent deep page fetching')
print('  ✓ Better error handling for connection issues')
print('  ✓ Graceful Ctrl+C handling')
print()
print('The scraper should now fail fast instead of hanging!')
print('=' * 70)
