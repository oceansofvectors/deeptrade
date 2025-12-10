# Common Crawl Financial News Scraper

Python tool to query and download financial news articles from the Common Crawl dataset.

## Overview

This scraper allows you to:
- Query Common Crawl's index for financial news websites
- Download article content from WARC archives
- Extract title, URL, and article text
- Save results to JSON or CSV format

## Supported Financial News Sites

The scraper includes built-in support for major financial news domains:

- Bloomberg (bloomberg.com)
- Reuters (reuters.com)
- Wall Street Journal (wsj.com)
- Financial Times (ft.com)
- The Economist (economist.com)
- Forbes (forbes.com)
- MarketWatch (marketwatch.com)
- Investing.com (investing.com)
- Yahoo Finance (finance.yahoo.com)
- CNBC (cnbc.com)
- Investopedia (investopedia.com)
- Nasdaq (nasdaq.com)

You can also scrape from any other domain available in Common Crawl.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Command Line Usage

Scrape articles using default settings (most recent index, all financial domains):

```bash
python common_crawl_scraper.py
```

Scrape from specific domains:

```bash
python common_crawl_scraper.py --domains reuters.com bloomberg.com --max-per-domain 20
```

Use a specific Common Crawl index:

```bash
python common_crawl_scraper.py --index CC-MAIN-2024-10
```

Save to CSV:

```bash
python common_crawl_scraper.py --output results.csv --format csv
```

### Python API Usage

```python
from common_crawl_scraper import CommonCrawlScraper, save_articles_to_json

# Create scraper
scraper = CommonCrawlScraper(max_articles_per_domain=50)

# Scrape from all financial domains
articles = scraper.scrape_all_domains()

# Save results
save_articles_to_json(articles, 'financial_news.json')

# Access article data
for article in articles:
    print(f"Title: {article.title}")
    print(f"URL: {article.url}")
    print(f"Content: {article.content[:200]}...")
    print(f"Domain: {article.domain}")
    print(f"Crawl Date: {article.crawl_date}")
    print("-" * 60)
```

### Scrape Single Domain

```python
scraper = CommonCrawlScraper(max_articles_per_domain=10)

# Get available indexes
indexes = scraper.get_available_indexes()
latest_index = indexes[0]

# Scrape Reuters
articles = scraper.scrape_domain('reuters.com', latest_index)
```

### Scrape Specific Domains

```python
scraper = CommonCrawlScraper(max_articles_per_domain=25)

# Custom domain list
domains = ['bloomberg.com', 'wsj.com', 'ft.com']

articles = scraper.scrape_all_domains(domains=domains)
```

## Examples

Run the example scripts:

```bash
# Interactive examples
python example_usage.py

# Run specific example
python example_usage.py 1  # Scrape single domain
python example_usage.py 2  # Scrape multiple domains
python example_usage.py 3  # Use specific index
python example_usage.py 4  # Analyze articles
python example_usage.py 5  # Custom domain list
python example_usage.py all  # Run all examples
```

## Output Format

### JSON Format

```json
[
  {
    "url": "https://www.reuters.com/article/...",
    "title": "Market Analysis: Tech Stocks Rally",
    "content": "Full article text...",
    "domain": "reuters.com",
    "crawl_date": "20240315120000",
    "fetch_date": "2024-03-20T10:30:00"
  }
]
```

### CSV Format

Columns: `url`, `title`, `content`, `domain`, `crawl_date`, `fetch_date`

## How It Works

1. **Index Query**: Queries Common Crawl's CDX index API to find URLs from specified domains
2. **WARC Fetch**: Downloads specific WARC records from Common Crawl's S3 bucket using HTTP range requests
3. **Parsing**: Extracts HTML content from WARC format
4. **Extraction**: Uses BeautifulSoup to extract article title and content
5. **Validation**: Filters out non-article pages (too short, no proper content structure)

## Common Crawl Indexes

Common Crawl creates new indexes approximately monthly. Each index contains billions of web pages. The scraper automatically uses the most recent index unless specified.

Index naming format: `CC-MAIN-YYYY-WW` (year and week number)

Example indexes:
- CC-MAIN-2024-10 (March 2024)
- CC-MAIN-2024-05 (January 2024)
- CC-MAIN-2023-50 (December 2023)

## Configuration Options

### CommonCrawlScraper Parameters

- `max_articles_per_domain`: Maximum number of articles to fetch per domain (default: 100)

### Command Line Arguments

- `--index`: Specific Common Crawl index name (default: most recent)
- `--domains`: Space-separated list of domains to scrape
- `--max-per-domain`: Maximum articles per domain (default: 50)
- `--output`: Output file path (default: financial_news.json)
- `--format`: Output format - json or csv (default: json)

## Rate Limiting

The scraper includes built-in rate limiting:
- 0.5 second delay between article fetches
- 1 second delay between domains
- Prevents overwhelming Common Crawl servers

## Error Handling

The scraper handles:
- Network timeouts and connection errors
- Malformed WARC records
- Invalid HTML parsing
- Missing article content
- HTTP errors

Errors are logged but don't stop the scraping process.

## Performance Considerations

- Each WARC fetch is ~10-50 KB (compressed)
- Expect 1-2 seconds per article (including network latency)
- Scraping 100 articles takes approximately 2-5 minutes
- Use smaller `max_articles_per_domain` for faster results

## Data Quality

Not all scraped content will be perfect:
- Some pages may be category pages or homepages (filtered by content length)
- Paywalled articles may have limited content
- HTML parsing depends on site structure
- Content validation filters out pages < 200 characters

## Limitations

- Only includes content available in Common Crawl (respects robots.txt)
- Historical data only (not real-time)
- Some paywalled content may be incomplete
- Scraping speed limited by network and rate limiting

## Advanced Usage

### Custom Article Extraction

Extend the `extract_article_content` method for domain-specific parsing:

```python
class CustomScraper(CommonCrawlScraper):
    def extract_article_content(self, html, url):
        # Custom parsing logic for specific sites
        if 'bloomberg.com' in url:
            # Bloomberg-specific extraction
            pass
        return super().extract_article_content(html, url)
```

### Filtering Articles

```python
articles = scraper.scrape_all_domains()

# Filter by keyword
tech_articles = [a for a in articles if 'technology' in a.content.lower()]

# Filter by date
from datetime import datetime
recent_articles = [
    a for a in articles
    if a.crawl_date > '20240101000000'
]

# Filter by length
long_articles = [a for a in articles if len(a.content) > 1000]
```

## Troubleshooting

**No articles found:**
- Try a different index (older indexes may have different coverage)
- Check if the domain is actually in Common Crawl using their index search
- Increase `max_articles_per_domain`

**Extraction errors:**
- Some sites have complex HTML structures that may not parse correctly
- Try examining the raw HTML to understand the site structure
- Consider customizing the extraction logic for specific domains

**Slow scraping:**
- Reduce `max_articles_per_domain`
- Use fewer domains
- Common Crawl S3 fetches can be slow depending on network conditions

## Legal and Ethical Considerations

- Common Crawl data is publicly available
- Respect copyright and terms of service of original content
- Use for research, analysis, and non-commercial purposes
- Be mindful of the original publishers' rights

## References

- [Common Crawl](https://commoncrawl.org/)
- [Common Crawl Index API](https://index.commoncrawl.org/)
- [CC-NEWS Dataset](https://commoncrawl.org/2016/10/news-dataset-available/)

## License

This tool is for educational and research purposes.
