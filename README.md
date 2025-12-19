# 📚 MyStudyMate – Personalized RAG-Based Chatbot for Personal Academic Materials

MyStudyMate is a **Retrieval-Augmented Generation (RAG) web application** that allows users to upload PDF documents and interact with them through a conversational AI interface.  
It retrieves relevant content from documents and generates **faithful, context-aware answers**, making it ideal for **studying, revision, and academic assistance**.

The project also includes an **automatic RAG evaluation pipeline** using **DeepEval** to assess answer quality and hallucination control.

---

##  Features

-  Upload and index multiple PDF documents  
-  Semantic search using **FAISS** vector database  
-  Conversational AI powered by **LangChain + Ollama / Groq**  
-  HuggingFace sentence embeddings  
-  Persistent chat history  
-  Modern **Flask + Tailwind CSS** interface  
-  Automated RAG evaluation (Relevancy & Faithfulness)  
-  Multilingual responses (same language as the user)  
-  Answers strictly grounded in document content  

## Architecture 
<img width="5958" height="6830" alt="image1" src="https://github.com/user-attachments/assets/206481bb-d8c9-4144-a038-a85b96609ebd" />

## Key Components Interaction
<img width="1201" height="352" alt="image" src="https://github.com/user-attachments/assets/0e37f5ca-9dc5-4813-8b4d-1ba03576c266" />


## Home & PDF Processing Interface
<img width="945" height="413" alt="image" src="https://github.com/user-attachments/assets/446432a9-ecb3-445a-a0d4-71c21eeb0ff6" />

<p>This screenshot shows the main interface of MyStudyMate, where users can upload multiple PDF documents, process them into a vector database, and start an interactive conversation. The assistant provides document-grounded answers with source references, offering a smooth and intuitive study experience through a modern, dark-themed UI.</p>


## Faithfulness & Hallucination Control
<img width="945" height="406" alt="image" src="https://github.com/user-attachments/assets/8b3d2d5a-5712-413d-b3fa-8a8a1f116645" />

<p>This screenshot demonstrates MyStudyMate’s strict document-grounded behavior. When a user asks a question that is not covered in the uploaded PDFs, the system explicitly responds that the information is not mentioned in the content, ensuring high faithfulness and preventing hallucinated answers.</p>

## RAG Evaluation Results
<img width="573" height="259" alt="image" src="https://github.com/user-attachments/assets/5233e558-dfc6-4311-a56c-b83bb4248db4" />
<img width="767" height="259" alt="image" src="https://github.com/user-attachments/assets/d033addf-2b07-4a8f-b964-125335c9a195" />

<p>This screenshot shows the automatic evaluation of the RAG pipeline using DeepEval. The system achieves 100% pass rate, with perfect scores in Answer Relevancy and Faithfulness, demonstrating that the generated responses are both relevant to the user query and strictly grounded in the retrieved document context.</p>





