import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv
###
# Load environment variables (e.g., API keys)
load_dotenv()

# Initialize tools
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

search = DuckDuckGoSearchRun(name="Search")

# Streamlit interface
st.title("🔎 LangChain - Chat with search")
"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

# Sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# Initialize session state for storing chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

# Display conversation history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Input for user query
if prompt := st.chat_input(placeholder="What is machine learning?"):
    # Append the user input to the conversation history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Initialize the Groq-based language model
    llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)

    # Define the tools for the agent
    tools = [search, arxiv, wiki]

    # Initialize the agent with the tools and language model
    search_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        handle_parsing_errors=True
    )

    try:
        with st.chat_message("assistant"):
            # Setup the Streamlit callback handler for streaming
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

            # Provide both the user input and chat history to the agent
            response = search_agent.run({
                "input": prompt, 
                "chat_history": st.session_state.messages
            }, callbacks=[st_cb])

            # Append the assistant's response to the conversation history
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
    
    except Exception as e:
        # Handle any exceptions and provide feedback to the user
        st.session_state.messages.append({"role": "assistant", "content": "I encountered an issue while processing your request."})
        st.error(f"Error: {e}")
