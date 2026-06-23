Learn the fundamentals of Qdrant, a vector-first database, 
including data loading, similarity matching, natural language searching, and building a simple FastAPI interface. 

In this project, you will develop a solution for semantic search using Qdrant, 
using OpenAI's embeddings API to process data, perform similarity searches, 
and construct an interface that enables retrieval of data points through natural language queries and filtering techniques.

STAGES
1. Parse the dataset and load it into Qdrant
2. Find the entries that are the closest to an existing embedding
3. Use OpenAI's embedding model to transform natural langauge queries and find the most similar entries in the database
4. Augment the search with a payload filter to find more relevant entries
5. Write a simple FastAPI wrapper to interact with the database
