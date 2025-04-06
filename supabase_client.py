from supabase import Client
from typing import List, Dict, Any
from config import Config

class SupabaseClient:
    """Classe per interagire con Supabase."""
    
    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client
    
    async def retrieve_relevant_documentation(self, query_embedding: List[float], match_count: int = 5) -> str:
        """Recupera la documentazione pertinente da Supabase in base all'embedding."""
        try:
            result = self.supabase.rpc('match_site_pages', {
                'query_embedding': query_embedding,
                'match_count': match_count,
                'filter': {'source': 'mobiscroll_angular_ai_docs'}
            }).execute()
            
            if not result.data:
                return "Nessuna documentazione rilevante trovata."
            
            formatted_chunks = []
            for doc in result.data:
                chunk_text = f"""
                # {doc['title']}

                {doc['content']}
                """
                formatted_chunks.append(chunk_text)
            
            return "\n\n---\n\n".join(formatted_chunks)
        except Exception as e:
            print(f"Errore nel recupero dei documenti: {e}")
            return "Errore nel recupero della documentazione."
    async def list_documentation_pages(self) -> List[str]:
        """
        Retrieve a list of all available Pydantic AI documentation pages.
        
        Returns:
            List[str]: List of unique URLs for all documentation pages
        """
        try:
            # Query Supabase for unique URLs where source is pydantic_ai_docs
            result = self.supabase.from_('site_pages') \
                .select('url') \
                .eq('metadata->>source', 'mobiscroll_angular_ai_docs') \
                .execute()
            
            if not result.data:
                return []
                
            # Extract unique URLs
            urls = sorted(set(doc['url'] for doc in result.data))
            return urls
            
        except Exception as e:
            print(f"Error retrieving documentation pages: {e}")
            return []
    async def get_page_content(self,url: str) -> str:
        """
        Retrieve the full content of a specific documentation page by combining all its chunks.
        
        Args:
            ctx: The context including the Supabase client
            url: The URL of the page to retrieve
            
        Returns:
            str: The complete page content with all chunks combined in order
        """
        try:
            # Query Supabase for all chunks of this URL, ordered by chunk_number
            result = self.supabase.from_('site_pages') \
                .select('title, content, chunk_number') \
                .eq('url', url) \
                .eq('metadata->>source', 'mobiscroll_angular_ai_docs') \
                .order('chunk_number') \
                .execute()
            
            if not result.data:
                return f"No content found for URL: {url}"
                
            # Format the page with its title and all chunks
            page_title = result.data[0]['title'].split(' - ')[0]  # Get the main title
            formatted_content = [f"# {page_title}\n"]
            
            # Add each chunk's content
            for chunk in result.data:
                formatted_content.append(chunk['content'])
                
            # Join everything together
            return "\n\n".join(formatted_content)
            
        except Exception as e:
            print(f"Error retrieving page content: {e}")
            return f"Error retrieving page content: {str(e)}"
