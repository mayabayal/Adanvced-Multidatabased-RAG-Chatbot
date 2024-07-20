# app.py
import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.base import Tool
from langchain import hub
from langchain.agents import create_openai_tools_agent, AgentExecutor

# Streamlit settings
st.set_page_config(page_title="Chatbot with LangChain", page_icon="🤖")
st.title("Chatbot with LangChain")

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("OPENAI_API_KEY not found. Please set it in the .env file.")
else:
    st.success("OPENAI_API_KEY loaded successfully.")

# Initialize OpenAI LLM
llm = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo-0125", temperature=0)

# Wikipedia Tool
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

# WebBaseLoader and FAISS for LangSmith documentation
loader = WebBaseLoader("https://docs.smith.langchain.com/")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
vectordb = FAISS.from_documents(documents, OpenAIEmbeddings(openai_api_key=openai_api_key))
retriever = vectordb.as_retriever()

# Custom retriever tool function
def create_retriever_tool(retriever, name, description):
    return Tool(name=name, func=retriever.get_relevant_documents, description=description)

retriever_tool = create_retriever_tool(retriever, "langsmith_search", "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!")

# Arxiv Tool
def arxiv_query_run(query):
    arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    return arxiv_wrapper.run(query)

arxiv_tool = Tool(name="arxiv_search", func=arxiv_query_run, description="Search for academic papers on Arxiv.")

# Combine tools
tools = [wiki, arxiv_tool, retriever_tool]

# Load the prompt
prompt = hub.pull("hwchase17/openai-functions-agent")

# Create agent with tools and prompt
agent = create_openai_tools_agent(llm, tools, prompt)

# Create AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Streamlit interface for chatbot
st.write("Ask me anything!")

user_input = st.text_input("Your question:")

if user_input:
    with st.spinner("Thinking..."):
        response = agent_executor.invoke({"input": user_input})
        st.write(response)
