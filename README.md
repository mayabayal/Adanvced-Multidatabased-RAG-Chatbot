# Advanced RAG Q&A Application with LangChain

## Overview

This project demonstrates the creation of an advanced Retrieval Augmented Generation (RAG) Q&A application using LangChain. By integrating multiple data sources—Wikipedia, a custom website, and a research paper database (RIVE)—this application provides comprehensive answers by dynamically selecting the most relevant data source for each query.

## Highlights

- **Integration of Multiple Data Sources**: Creates wrappers for Wikipedia, a custom website, and RIVE, enabling interaction with these sources as distinct entities.
- **LangChain Agents**: Uses LangChain agents to dynamically select the appropriate tool based on user queries.
- **Combining Tools**: Combines multiple tools into a single list using Python and connects them with the chosen LLM model via `create_openai_tool_agent`.
- **Prompt Templates**: Utilizes prompt templates from LangChain’s Hub to guide LLM interactions.
- **Agent Execution**: Executes the agent and tools using `agent_executor` to retrieve information and provide comprehensive responses.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/advanced-rag-qa-app.git
   cd advanced-rag-qa-app
2. **Install Dependencies**:
   pip install -r requirements.txt
3. **Set Up Environment Variables**:
  Create a .env file in the root directory and add the following variables:
   - OPENAI_API_KEY=your_openai_api_key
   - WIKIPEDIA_API_KEY=your_wikipedia_api_key
   - CUSTOM_WEBSITE_URL=your_custom_website_url

**Usage**
Run the Application:
streamlit run app.py

**Interact with the Application**:
Open the provided URL in your browser to start querying the RAG Q&A application.

Project Structure
app.py: Main script to run the Streamlit app.
config.py: Configuration settings and environment variables.
langchain_utils.py: Utility functions for LangChain integration.
data_sources/: Contains wrappers for Wikipedia, custom website, and RIVE.
templates/: Prompt templates used for guiding LLM interactions.
requirements.txt: List of Python dependencies.
Contributing
If you would like to contribute to this project, please fork the repository and submit a pull request with your changes.

