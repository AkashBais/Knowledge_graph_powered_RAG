# Knowledge_graph_powered_RAG
## Retrieval Augmented Generation (RAG) Project
### Overview
This project is a Retrieval Augmented Generation (RAG) system designed to parse standard PDF documents. It helps parse your PDF documents and also enhance the knowledge in them with relevant WEB search . **The system is equipped with a Knowledge Graph backend, replacing the traditional vector store**, which enhances its ability to parse PDF documents[while retaining its structural information] and supplement its sections with relevant web search results.

### Features
**Knowledge Graph Backend**: Utilizes a Knowledge Graph for enhanced information retrieval above simple similarity search.  
**Summarization**: Provides concise and accurate summaries of the information gathered.  
**Enhanced Information Retrieval**: Supplements parsed document sections with relevant web search results for comprehensive answers.  
**Multi-Model type support**: It lets you initilize one of a Hugging Face LLM, Gimini as LLM, or A GGUF model to run on CPU based systems.  
### How It Works
**Document Input**: Users input their document of interest. The system parces it and extracts relevant headers and section text for those headers  
**Web Search**: The system performs relevant web searches to gather suplimental information for the headers.  
**Knowledge Graph Integration**: Utilizes the Knowledge Graph to parse and understand PDF documents for better retrival.  
**Information Synthesis**: Analyzes and summarizes the gathered information to provide a coherent response.  
**Response Generation**: Delivers a personalized and concise answer to the user’s query.  
### Use Cases
Interacting with your PDF documents   
General Knowledge Queries
