from  openai import AsyncOpenAI
from supabase import Client
from typing import List
import json

class OpenAIClient:
    """Classe per interagire con l'API OpenAI."""
    
    def __init__(self,  llm_model: str, llm_model_embeddings: str, openai_client: AsyncOpenAI, supabase_client: Client):
        self.llm_model = llm_model
        self.llm_model_embeddings = llm_model_embeddings
        self.openai_client = openai_client
        self.supabase = supabase_client
        self.instructions = """
            Sei un esperto di Mobiscroll per Angular - una collezione di componenti UI di cui hai accesso a tutta la documentazione, inclusi esempi, riferimenti API e 
            ad altre risorse per aiutarti a costruire bellissime web application.

            Il vostro unico compito è quello di fornire assistenza e non rispondete ad altre domande oltre a descrivere ciò che siete in grado di fare.

            Non chiedete all'utente prima di compiere un'azione, fatela e basta. Prima di rispondere alla domanda dell'utente, assicuratevi sempre di consultare la documentazione con gli strumenti forniti, 
            a meno che non l'abbiate già fatto.

            Quando si esamina la documentazione per la prima volta, iniziare sempre con RAG.
            Poi controllate sempre anche l'elenco delle pagine di documentazione disponibili e recuperate il contenuto delle pagine se può essere utile.

            Comunicate sempre all'utente quando non avete trovato la risposta nella documentazione o nell'URL giusto - siate onesti.
            """
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "retrieve_relevant_documentation",
                    "description": "Recupera i documenti più rilevanti per una query",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "La domanda o descrizione della funzionalità cercata"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_documentation_pages",
                    "description": "Restituisce un elenco di URL di tutte le pagine della documentazione",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_page_content",
                    "description": "Restituisce il contenuto testuale di una pagina specifica",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "URL della pagina di documentazione"
                            }
                        },
                        "required": ["url"]
                    }
                }
            }
        ]
    
    async def get_embedding(self, text: str) -> List[float]:
        """Ottiene un embedding vettoriale da OpenAI."""
        try:
            response = await self.openai_client.embeddings.create(
                model=self.llm_model_embeddings,  # Puoi cambiare il modello
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Errore nel recupero dell'embedding: {e}")
            return [0] * 768  # Vettore zero in caso di errore
    async def get_completion_with_tools(self, user_input: str) -> str:
        """Gestisce la logica completa di tool calling."""
        messages = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": user_input},
        ]

        response = await self.openai_client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            tools=self.tools,
            tool_choice="auto"
        )

        first_choice = response.choices[0]
        print(first_choice)

        if first_choice.finish_reason == "tool_calls":
            tool_calls = first_choice.message.tool_calls
            tool_outputs = []

            for tool in tool_calls:
                func_name = tool.function.name
                args = json.loads(tool.function.arguments)

                if func_name == "retrieve_relevant_documentation":
                    embedding = await self.get_embedding(args["query"])
                    tool_output = await self.supabase.retrieve_relevant_documentation(embedding)
                elif func_name == "list_documentation_pages":
                    tool_output = await self.supabase.list_documentation_pages()
                elif func_name == "get_page_content":
                    tool_output = await self.supabase.get_page_content(args["url"])
                else:
                    tool_output = f"Tool {func_name} non gestito."
                
                tool_outputs.append({
                    "tool_call_id": tool.id,
                    "name": func_name,
                    "content": str(tool_output)
                })

            follow_up_messages = messages + [{"role": "tool", **output} for output in tool_outputs]

            return await self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=follow_up_messages,
                stream=True
            )
        else:
            return first_choice.message.content
    async def get_completion(self, prompt: str) -> str:
        """Ottiene una risposta da OpenAI usando il completamento."""
        try:
            return await self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": self.instructions},
                    {"role": "user", "content": prompt},
                ],
                stream=True
            )
            # return response.choices[0].message.content
        except Exception as e:
            print(f"Errore nel recupero della risposta: {e}")
            return "Errore durante il completamento."
