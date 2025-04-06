import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter, DomainFilter

async def main():
    # Configure a 2-level deep crawl
    filter_chain = FilterChain([
        # Only follow URLs with specific patterns
        URLPatternFilter(patterns=[r"^https://mobiscroll\.com/docs/angular/.*"]),

        # Only crawl specific domains
        DomainFilter(
            allowed_domains=["mobiscroll.com"],
            blocked_domains=["status.mobiscroll.com", "blog.mobiscroll.com", "help.mobiscroll.com", "download.mobiscroll.com", "forum.mobiscroll.com"]
        ),
    ])
    config = CrawlerRunConfig(
        deep_crawl_strategy=BFSDeepCrawlStrategy(
            max_depth=3, 
            filter_chain=filter_chain,
            include_external=False
        ),
        scraping_strategy=LXMLWebScrapingStrategy(),
        verbose=False
    )

    

    async with AsyncWebCrawler() as crawler:
        results = await crawler.arun("https://mobiscroll.com/docs/angular", config=config)

        print(f"Crawled {len(results)} pages in total")

        # Access individual results
        for result in results[:3]:  # Show first 3 results
            print(f"URL: {result.url}")
            print(f"Depth: {result.metadata.get('depth', 0)}")

if __name__ == "__main__":
    asyncio.run(main())