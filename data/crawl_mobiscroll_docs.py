import os
import json
import asyncio
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter, DomainFilter
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from openai import AsyncOpenAI
from supabase import create_client, Client

load_dotenv()

# Initialize OpenAI and Supabase clients
openai_client = AsyncOpenAI(base_url="http://localhost:11434/v1",api_key="ollama")
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif '\n\n' in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = max(start + 1, end)

    return chunks

async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """Extract title and summary using GPT-4."""
    system_prompt = """Sei una intelligenza artificiale che estrae titoli e sommari dai pezzi di documentazione.
    Restituisci un oggetto JSON con le chiavi “title” e “summary”.
    Per "title": Se sembra l'inizio di un documento, estrarre il titolo. Se si tratta di un pezzo centrale, ricavare un titolo descrittivo.
    Per "summary": creare un sommario conciso dei punti principali di questo pezzo.
    Mantenere sia "title" che "summary" concisi ma informativi."""
    
    try:
        response = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "llama3.2"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}  # Send first 1000 chars for context
            ],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting title and summary: {e}")
        return {"title": "Error processing title", "summary": "Error processing summary"}

async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model=os.getenv("LLM_MODEL_EMBEDDINGS", "nomic-embed-text"),
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 768  # Return zero vector on error

def get_first_string(value):
    if isinstance(value, list):
        return value[0] if value and isinstance(value[0], str) else ""
    return value if isinstance(value, str) else ""


async def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    """Process a single chunk of text."""
    # Get title and summary
    extracted = await get_title_and_summary(chunk, url)
    
    title = get_first_string(extracted.get("title", "")).strip()
    summary = get_first_string(extracted.get("summary", "")).strip()

    if not title or not summary:
        print(f"Skipping chunk {chunk_number} — missing title or summary")
        return None
    
    # Get embedding
    embedding = await get_embedding(chunk)
    
    # Create metadata
    metadata = {
        "source": "mobiscroll_angular_ai_docs",
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path
    }
    
    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted['title'],
        summary=extracted['summary'],
        content=chunk,  # Store the original chunk content
        metadata=metadata,
        embedding=embedding
    )

async def insert_chunk(chunk: ProcessedChunk):
    """Insert a processed chunk into Supabase."""
    try:
        data = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding
        }
        
        result = supabase.table("site_pages").insert(data).execute()
        print(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
        return result
    except Exception as e:
        print(f"Error inserting chunk: {e}")
        return None

async def process_and_store_document(url: str, markdown: str):
    """Process a document and store its chunks in parallel."""
    # Split into chunks
    chunks = chunk_text(markdown)
    
    # Process chunks in parallel
    # tasks = [
    #     process_chunk(chunk, i, url) 
    #     for i, chunk in enumerate(chunks)
    # ]
    # processed_chunks = await asyncio.gather(*tasks)

    semaphore = asyncio.Semaphore(5)

    async def limited_process_chunk(chunk, chunk_number, url):
        async with semaphore:
            return await process_chunk(chunk, chunk_number, url)

    tasks = [
        limited_process_chunk(chunk, i, url)
        for i, chunk in enumerate(chunks)
    ]
    processed_chunks = await asyncio.gather(*tasks)
    
    # Filtra quelli validi (non None)
    valid_chunks = [chunk for chunk in processed_chunks if chunk is not None]
    # Store chunks in parallel
    insert_tasks = [
        insert_chunk(chunk) 
        for chunk in valid_chunks
    ]
    await asyncio.gather(*insert_tasks)

async def crawl_deep(url: str):
    """Crawl multiple URLs in parallel with a concurrency limit."""
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    filter_chain = FilterChain([
        URLPatternFilter(patterns=[r"^https://mobiscroll\.com/docs/angular/.*"]),
        DomainFilter(
            allowed_domains=["mobiscroll.com"],
            blocked_domains=["status.mobiscroll.com", "blog.mobiscroll.com", "help.mobiscroll.com", "download.mobiscroll.com", "forum.mobiscroll.com"]
        ),
    ])
    crawl_config = CrawlerRunConfig(
        deep_crawl_strategy=BFSDeepCrawlStrategy(
            max_depth=3, 
            filter_chain=filter_chain,
            include_external=False
        ),
        cache_mode=CacheMode.BYPASS,
        scraping_strategy=LXMLWebScrapingStrategy(),
        verbose=True,
        stream=True
    )

    # Create the crawler instance
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()
    
    try:
        async for result in await crawler.arun(url, config=crawl_config):
        # Process each result as it becomes available
            await process_and_store_document(result.url, result.markdown.raw_markdown)
    finally:
        await crawler.close()

async def main():
    # Get URLs from Pydantic AI docs
    url = "https://mobiscroll.com/docs/angular"
    await crawl_deep(url)

if __name__ == "__main__":
    asyncio.run(main())