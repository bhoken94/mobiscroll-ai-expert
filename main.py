import asyncio
import streamlit as st
from config import Config
from supabase_client import SupabaseClient
from openai_client import OpenAIClient

st.markdown(
    """
    <style>
        .stProgress .st-bo {
            background-color: #00a0dc;
        }
    </style>
""",
    unsafe_allow_html=True,
)

class App:
    """Classe principale per l'applicazione."""
    
    def __init__(self):
        # Configura le dipendenze
        self.config = Config()
        self.config.setup_logfire()
        self.supabase_client = SupabaseClient(self.config.setup_supabase())
        self.openai_client = OpenAIClient(Config.LLM_MODEL, Config.LLM_MODEL_EMBEDDINGS, self.config.setup_openai(), self.supabase_client)
        # self.chat_session = ChatSession()

    
    async def run(self, user_input: str):
        """Esegui la logica dell'applicazione."""

        with st.chat_message("user"):
            st.markdown(user_input)

        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.spinner("Sto pensando...", show_time=True):
            # Ottieni la risposta dal modello OpenAI
            response_stream = await self.openai_client.get_completion_with_tools(user_input)
        # Aggiungi la risposta alla sessione
        with st.chat_message("assistant"):
            response_text = ""
            response_placeholder = st.empty()
            # Controllo se la risposta è una stringa completa o viene ricevuta tramite streaming
            if isinstance(response_stream, str):  # Caso di una risposta completa
                response_text = response_stream
                response_placeholder.markdown(response_text)
            else:  # Caso di risposta in streaming
                async for chunk in response_stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        delta = chunk.choices[0].delta.content
                        response_text += delta
                        response_placeholder.markdown(response_text)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response_text})

# Funzione principale per Streamlit
async def main():
    st.title("Chat con Mobiscroll Angular")
    st.write("Fai una domanda sulle API di Mobiscroll per Angular e ricevi una risposta.")

    app = App()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input dell'utente
    user_input = st.chat_input("Quali domande hai su Mobiscroll?")
    
    if user_input:
        await app.run(user_input)

if __name__ == "__main__":
    asyncio.run(main())
