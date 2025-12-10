# CommonCrawl Scraper - Fixes Applied

## Problem

The CommonCrawl index API at `index.commoncrawl.org` was experiencing connection issues:

1. **Connection Refused Errors** - Port 443 not accepting connections
2. **Timeout Errors** - Connections hanging without response
3. **404 Errors** - Some index endpoints returning "Not Found"

## Root Cause

The primary CommonCrawl index server endpoint (`https://index.commoncrawl.org/collinfo.json`) appears to be down or unreachable. However, individual per-index endpoints (e.g., `https://index.commoncrawl.org/CC-MAIN-2024-10-index`) may still be accessible.

## Solution Implemented

### 1. Added CDX Toolkit Integration

**What is CDX Toolkit?**
- A Python library that provides robust access to CommonCrawl's CDX (Capture Index) data
- Handles fallback between multiple index endpoints automatically
- More resilient to individual endpoint failures

**Changes Made:**
- Added `cdx-toolkit>=0.9.31` to `requirements.txt`
- Integrated CDX Toolkit as an optional dependency with graceful degradation
- Added automatic fallback mechanism when direct API fails

### 2. Improved Error Handling

**Enhanced Exception Handling:**
```python
except (requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.HTTPError) as e:
    # Automatically fallback to CDX Toolkit
```

**Fallback Strategy:**
1. Try direct API first (backward compatible)
2. If direct API fails → automatically use CDX Toolkit
3. If CDX Toolkit unavailable → return empty results with clear error message

### 3. Hardcoded Index List Fallback

Added known recent CommonCrawl indexes as fallback when `collinfo.json` is unavailable:
```python
'CC-MAIN-2025-43',
'CC-MAIN-2025-40',
'CC-MAIN-2024-51',
'CC-MAIN-2024-46',
'CC-MAIN-2024-42',
```

### 4. Multiple URL Pattern Attempts

CDX Toolkit tries multiple URL patterns to maximize chances of finding results:
- `domain/*`
- `*.domain/*`
- `http://domain/*`
- `https://domain/*`

## How to Use

### Installation

```bash
pip install -r requirements.txt
```

This now includes `cdx-toolkit` which will be automatically used as fallback.

### Usage (No Changes Required!)

The scraper works exactly as before - the fallback is automatic:

```python
from common_crawl_scraper import CommonCrawlScraper

scraper = CommonCrawlScraper(max_articles_per_domain=50)
articles = scraper.scrape_all_domains()
```

### Force CDX Toolkit (Optional)

You can force the use of CDX Toolkit if you know the direct API is down:

```python
scraper = CommonCrawlScraper(max_articles_per_domain=50, use_cdx_toolkit=True)
```

## Testing

Run the quick test to verify the scraper is working:

```bash
python test_quick.py
```

Expected output:
```
✓ CDX Toolkit available: True
✓ SUCCESS: Found X records
✓ Scraper is working with CDX Toolkit fallback!
```

## Performance Notes

- **CDX Toolkit** may be slower than direct API as it queries multiple indexes
- Rate limiting is built-in to avoid overwhelming CommonCrawl servers
- For best performance, specify a recent index explicitly

## What Changed in Code

### Files Modified:
1. **requirements.txt** - Added `cdx-toolkit>=0.9.31`
2. **common_crawl_scraper.py**:
   - Added CDX Toolkit import and initialization
   - Added `search_domain_cdx_toolkit()` method
   - Enhanced `search_domain()` with automatic fallback
   - Improved error handling in `get_available_indexes()`
   - Added multiple exception catches for robustness

### Files Created:
1. **test_quick.py** - Quick test script
2. **FIXES_APPLIED.md** - This document

## Compatibility

- ✅ Backward compatible - existing code works without changes
- ✅ Graceful degradation - works even if CDX Toolkit not installed
- ✅ No breaking changes to API

## Future Recommendations

1. **Monitor CommonCrawl Status**: Check https://status.commoncrawl.org for service updates
2. **Consider Caching**: Cache index lists to reduce API calls
3. **Rate Limiting**: Current implementation has conservative rate limits - adjust if needed
4. **Alternative Approaches**:
   - Direct download of index files from S3
   - Use CommonCrawl's Columnar Index (Parquet format) for large-scale queries

## Summary

The scraper is now **production-ready** with automatic fallback to CDX Toolkit when the direct CommonCrawl API is unavailable. No user action required - the fallback happens automatically!
