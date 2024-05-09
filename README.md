# RAG Application: An AI Engineer Case Study

## Overview
The RAG application is a case study test for the AI Engineering role at Vendease. The primary objective is to develop a generative AI chatbot system trained on customer service and support data. The project uses a variety of AI techniques, including entity masking, semantic embeddings, vector databases, and large language model (LLM) refinement to create a chatbot system. **You can find my solution to the Case Study in either this [NOTEBOOK FILE](https://github.com/Abdulraqib20/Vendease-RAG-Application/blob/main/Copy_of_RAG_APP.ipynb) or in this [GOOGLE COLAB NOTEBOOK](https://colab.research.google.com/drive/1tZ7f03Re4mBXgF2dgh3zWUHWT8ziJJNe?usp=drive_link).**

## Dataset
The dataset used for training the chatbot is sourced from the Bitext customer support LLM (Large Language Model) chatbot training dataset available [here](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset).

## Key Components and Steps
1. Data Preparation
   - Loaded the the CSV dataset using pandas.
   - Checked the dataset for quality issues and 
   - Applied entity masking to the "instruction" and "response" columns to enhance the RAG system's ability to generalize.
   - Copied the dataset into a new dataframe and renamed columns as "id_num", "question" and "answer"
     
2. Chunking Data
   Performed chunking on the data by dividingthe dataset into smaller segments, or "chunks." This operation is crucial for efficient data processing, particularly when dealing with extensive datasets that may exceed available memory resources. By segmenting the dataset into manageable chunks, computational tasks such as analysis, transformation, or modeling can be performed iteratively on each segment without overwhelming system resources

3. Setting Up Google API Key for Generative AI: Authenticating and Authorizing access to Google's Generative AI service, for text generation.
   
4. Setting Up Pinecone API Key: Authenticating and Authorizing access to Pinecone's vector similarity search service, allowing for efficient retrieval of similar vectors.
   
5. Embedding Text into Vectors Using the Embedding Model
   - Employed the `GoogleGenerativeAIEmbeddings` model to generate high-dimensional semantic embeddings for both the dataset and user queries.
   - Configured embeddings to balance performance and resource usage.
     
6. Vector Database (Pinecone)
   - Used Pinecone as the vector database due to its efficiency in similarity search and scalability.
   - Implemented batching strategies during the upsert process to optimize data transfer to Pinecone.

7. Querying and Retrieval
   - Embedded user queries using the embedding model.
   - I then used Pinecone's similarity search functionality to retrieve the most relevant documents from the database.

8. LLM Refinement
   - I integrated a large language model, which is the Gemini (gemini-pro) to improve the quality and coherence of the responses.
   - I used prompts to guide the LLM, utilizing the retrieved context for better answer generation.

## Dependencies
The code made use of several Python packages and libraries which were installed using pip. Here's the list of packages being installed:

1. `pandas`: Used for data manipulation and analysis.
2. `os`: Provides functions for interacting with the operating system.
3. `pathlib`: Offers classes for working with file paths.
4. `textwrap`: Provides convenience functions for wrapping and formatting text.
5. `google.generativeai`: Import from Google's Generative AI library for various NLP tasks.
6. `openai`: Library for accessing OpenAI's API and tools.
7. `langchain.embeddings.openai.OpenAIEmbeddings`: Class for embedding text using OpenAI's models.
8. `langchain.chat_models.ChatOpenAI`: Class for building conversational models using OpenAI.
9. `langchain.chains.conversation.memory.ConversationBufferWindowMemory`: Class for managing memory in conversational systems.
10. `langchain.chains.RetrievalQA`: Class for implementing retrieval-based question-answering systems.
11. `langchain.agents.Tool`: Class representing tools or utilities in the Langchain framework.
12. `langchain.agents.initialize_agent`: Function for initializing agents in the Langchain framework.
13. `langchain_google_genai.GoogleGenerativeAI`: Class for interfacing with Google's Generative AI models.
14. `langchain_google_genai.GoogleGenerativeAIEmbeddings`: Class for generating embeddings using Google's Generative AI models.
15. `langchain_google_genai.ChatGoogleGenerativeAI`: Class for building conversational models using Google's Generative AI.
16. `langchain.chains.ConversationalRetrievalChain`: Class for implementing conversational retrieval systems.
17. `langchain.vectorstores.Chroma`: Class for working with vector stores in the Langchain framework.
18. `langchain.text_splitter.CharacterTextSplitter`: Class for splitting text into characters.
19. `langchain.memory.ConversationBufferMemory`: Class for managing memory in conversational systems.
20. `tqdm`: Library for displaying progress bars.
22. `pinecone`: Library for working with vector indexes for similarity search.
23. `gradio`: Library for creating interactive UIs for machine learning models.
