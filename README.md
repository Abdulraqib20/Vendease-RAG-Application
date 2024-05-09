# RAG Application: An AI Engineer Case Study

## Overview
The RAG application is a case study test for the AI Engineering role at Vendease. The primary objective is to develop a generative AI chatbot system trained on customer service and support data. The project uses a variety of AI techniques, including entity masking, semantic embeddings, vector databases, and large language model (LLM) refinement to create a chatbot system.

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
