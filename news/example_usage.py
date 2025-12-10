"""
Example usage of Common Crawl Financial News Scraper
"""

from common_crawl_scraper import CommonCrawlScraper, save_articles_to_json, save_articles_to_csv
import json


def example_1_scrape_single_domain():
    """Example: Scrape articles from a single domain"""
    print("Example 1: Scraping Reuters from most recent index")
    print("=" * 60)

    scraper = CommonCrawlScraper(max_articles_per_domain=10)

    # Get latest index
    indexes = scraper.get_available_indexes()
    if not indexes:
        print("No indexes available")
        return

    latest_index = indexes[0]
    print(f"Using index: {latest_index}\n")

    # Scrape Reuters
    articles = scraper.scrape_domain('reuters.com', latest_index)

    print(f"\nScraped {len(articles)} articles from Reuters")

    if articles:
        print("\nFirst article:")
        print(f"Title: {articles[0].title}")
        print(f"URL: {articles[0].url}")
        print(f"Content preview: {articles[0].content[:200]}...")

        # Save to JSON
        save_articles_to_json(articles, 'reuters_articles.json')


def example_2_scrape_multiple_domains():
    """Example: Scrape from multiple specific domains"""
    print("\nExample 2: Scraping multiple domains")
    print("=" * 60)

    scraper = CommonCrawlScraper(max_articles_per_domain=5)

    # Scrape from specific domains
    domains = ['bloomberg.com', 'cnbc.com', 'marketwatch.com']

    articles = scraper.scrape_all_domains(domains=domains)

    print(f"\nTotal articles scraped: {len(articles)}")

    # Group by domain
    by_domain = {}
    for article in articles:
        by_domain.setdefault(article.domain, []).append(article)

    print("\nArticles per domain:")
    for domain, domain_articles in by_domain.items():
        print(f"  {domain}: {len(domain_articles)} articles")

    # Save to CSV
    if articles:
        save_articles_to_csv(articles, 'multiple_domains.csv')


def example_3_scrape_specific_index():
    """Example: Scrape from a specific Common Crawl index"""
    print("\nExample 3: Using a specific index")
    print("=" * 60)

    scraper = CommonCrawlScraper(max_articles_per_domain=5)

    # List available indexes
    indexes = scraper.get_available_indexes()
    print(f"Available indexes (first 10):")
    for i, idx in enumerate(indexes[:10], 1):
        print(f"  {i}. {idx}")

    # Use a specific index (e.g., second most recent)
    if len(indexes) > 1:
        index_to_use = indexes[1]
        print(f"\nUsing index: {index_to_use}")

        articles = scraper.scrape_domain('wsj.com', index_to_use)

        print(f"\nScraped {len(articles)} articles from WSJ")

        if articles:
            save_articles_to_json(articles, 'wsj_specific_index.json')


def example_4_analyze_articles():
    """Example: Analyze scraped articles"""
    print("\nExample 4: Analyzing scraped articles")
    print("=" * 60)

    scraper = CommonCrawlScraper(max_articles_per_domain=20)

    # Scrape from financial news sites
    articles = scraper.scrape_all_domains(
        domains=['bloomberg.com', 'reuters.com']
    )

    if not articles:
        print("No articles scraped")
        return

    print(f"\nAnalysis of {len(articles)} articles:")
    print("-" * 60)

    # Title length statistics
    title_lengths = [len(article.title) for article in articles]
    print(f"Average title length: {sum(title_lengths) / len(title_lengths):.1f} characters")

    # Content length statistics
    content_lengths = [len(article.content) for article in articles]
    print(f"Average content length: {sum(content_lengths) / len(content_lengths):.0f} characters")
    print(f"Shortest article: {min(content_lengths)} characters")
    print(f"Longest article: {max(content_lengths)} characters")

    # Common words in titles (simple analysis)
    all_title_words = []
    for article in articles:
        words = article.title.lower().split()
        all_title_words.extend(words)

    from collections import Counter
    word_counts = Counter(all_title_words)
    print(f"\nMost common words in titles:")
    for word, count in word_counts.most_common(10):
        if len(word) > 3:  # Skip short words
            print(f"  {word}: {count}")

    # Save analysis
    analysis = {
        'total_articles': len(articles),
        'avg_title_length': sum(title_lengths) / len(title_lengths),
        'avg_content_length': sum(content_lengths) / len(content_lengths),
        'min_content_length': min(content_lengths),
        'max_content_length': max(content_lengths),
        'articles_by_domain': {
            domain: len([a for a in articles if a.domain == domain])
            for domain in set(a.domain for a in articles)
        }
    }

    with open('analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)

    print("\nAnalysis saved to analysis.json")


def example_5_custom_domain_list():
    """Example: Scrape from a custom list of domains"""
    print("\nExample 5: Custom domain list")
    print("=" * 60)

    # You can scrape from any domains, not just the predefined ones
    custom_domains = [
        'seekingalpha.com',
        'barrons.com',
        'fool.com',
        'zacks.com'
    ]

    scraper = CommonCrawlScraper(max_articles_per_domain=10)

    articles = scraper.scrape_all_domains(domains=custom_domains)

    print(f"\nScraped {len(articles)} articles from custom domains")

    if articles:
        save_articles_to_json(articles, 'custom_domains.json')


if __name__ == "__main__":
    import sys

    print("Common Crawl Financial News Scraper - Examples")
    print("=" * 60)
    print("\nAvailable examples:")
    print("  1. Scrape single domain (Reuters)")
    print("  2. Scrape multiple domains")
    print("  3. Use specific index")
    print("  4. Analyze scraped articles")
    print("  5. Custom domain list")
    print("  all. Run all examples")

    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\nEnter example number (1-5) or 'all': ").strip()

    examples = {
        '1': example_1_scrape_single_domain,
        '2': example_2_scrape_multiple_domains,
        '3': example_3_scrape_specific_index,
        '4': example_4_analyze_articles,
        '5': example_5_custom_domain_list,
    }

    if choice == 'all':
        for func in examples.values():
            func()
            print("\n")
    elif choice in examples:
        examples[choice]()
    else:
        print(f"Invalid choice: {choice}")
