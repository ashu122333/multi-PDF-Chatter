# multi-PDF-Chatter
The Multi-PDF Chatter project is a powerful question-answering system that allows users to upload PDF documents and interact with the content through natural language queries. The system is built using Streamlit for a user-friendly web interface, where users can upload multiple PDFs and ask questions related to the content.



Key features of the project include:

PDF Text Extraction: The uploaded PDFs are processed to extract text using reliable libraries, enabling efficient text parsing for question answering.
Embedding Generation: The text content from the PDFs is converted into high-dimensional vector embeddings using the ggrn/e5-small-v2 model, which helps in creating a semantic representation of the text.
Vector and Keyword Search: The system utilizes Pinecone to store and index the vector embeddings, enabling fast and accurate retrieval of relevant information. In addition to vector-based search, the project integrates BM25Encoder for keyword-based search, providing users with a comprehensive search experience.
Question Answering: For answering user queries, the project employs the Gemma2-9b-it model, a GroQ-based model that generates precise answers by leveraging both the vector search and keyword search results.
LangChain: LangChain is used to chain together various components of the system, including embedding generation, document retrieval, and answer generation, creating a seamless and efficient pipeline.
This project provides an interactive way to query large sets of documents, ensuring that users can quickly get relevant answers to their questions based on the uploaded PDFs.
