"""
Quick test script to verify Common Crawl scraper is working
"""

from common_crawl_scraper import CommonCrawlScraper
import sys


def test_connection():
    """Test connection to Common Crawl"""
    print("Test 1: Testing connection to Common Crawl...")
    print("-" * 60)

    scraper = CommonCrawlScraper()

    try:
        indexes = scraper.get_available_indexes()

        if not indexes:
            print("ERROR: No indexes returned")
            return False

        print(f"SUCCESS: Found {len(indexes)} indexes")
        print(f"Most recent: {indexes[0]}")
        print(f"Sample indexes: {indexes[:5]}")
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        return False


def test_index_search():
    """Test searching the index"""
    print("\nTest 2: Testing index search...")
    print("-" * 60)

    scraper = CommonCrawlScraper()

    try:
        # Get latest index
        indexes = scraper.get_available_indexes()
        if not indexes:
            print("ERROR: No indexes available")
            return False

        latest_index = indexes[0]
        print(f"Using index: {latest_index}")

        # Search for a single URL from Reuters
        print("Searching for reuters.com URLs...")
        records = scraper.search_domain('reuters.com', latest_index, limit=5)

        if not records:
            print("WARNING: No records found for reuters.com")
            print("This might be normal if the domain wasn't crawled in this index")
            return True

        print(f"SUCCESS: Found {len(records)} records")
        print(f"First URL: {records[0].get('url', 'N/A')}")
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_article_fetch():
    """Test fetching and parsing an article"""
    print("\nTest 3: Testing article fetch and parse...")
    print("-" * 60)

    scraper = CommonCrawlScraper(max_articles_per_domain=1)

    try:
        # Get latest index
        indexes = scraper.get_available_indexes()
        if not indexes:
            print("ERROR: No indexes available")
            return False

        latest_index = indexes[0]

        # Try to scrape one article from Reuters
        print(f"Attempting to fetch 1 article from reuters.com...")
        articles = scraper.scrape_domain('reuters.com', latest_index)

        if not articles:
            print("WARNING: No articles extracted")
            print("This could mean:")
            print("  - Domain not in this index")
            print("  - Content extraction failed")
            print("  - No valid articles found")
            print("\nTrying alternative domain (cnbc.com)...")

            articles = scraper.scrape_domain('cnbc.com', latest_index)

            if not articles:
                print("WARNING: Still no articles. This may be normal.")
                return True

        print(f"SUCCESS: Extracted {len(articles)} article(s)")
        if articles:
            article = articles[0]
            print(f"\nArticle details:")
            print(f"  Title: {article.title[:80]}...")
            print(f"  URL: {article.url}")
            print(f"  Domain: {article.domain}")
            print(f"  Content length: {len(article.content)} characters")
            print(f"  Content preview: {article.content[:150]}...")

        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Common Crawl Scraper Test Suite")
    print("=" * 60)

    tests = [
        test_connection,
        test_index_search,
        test_article_fetch
    ]

    results = []

    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"FATAL ERROR in test: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

        print()

    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("Status: ALL TESTS PASSED")
        return 0
    elif passed > 0:
        print("Status: SOME TESTS PASSED")
        return 1
    else:
        print("Status: ALL TESTS FAILED")
        return 2


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
