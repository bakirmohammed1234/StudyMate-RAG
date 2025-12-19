import os
import pickle
from flask import Flask, render_template, request, flash, redirect, url_for, session

# Tes imports LangChain
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
# from langchain_google_genai import ChatGoogleGenerativeAI
from werkzeug.utils import secure_filename

# Load env
load_dotenv()
gr_TOKEN = os.getenv("gr_TOKEN")
# Gemini_API_KEY = os.getenv("GEMINI_API_KEY")

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "super_secret_key")

UPLOAD_FOLDER = 'uploads'
VECTORSTORE_PATH = "faiss_store.pkl"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- MÉMOIRE GLOBALE (Pour l'historique du chat) ---
# Dans une vraie app de prod, on utiliserait une base de données.

CHAT_HISTORY = [] 

print("Chargement du modèle d'embedding...")
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Patch manuel
embedder.query_encode_kwargs = {}
llm = ChatOllama(
    model="gpt-oss:120b-cloud", # Remplacez par le nom exact du modèle (ex: "mistral", "gemma2", etc.)
    temperature=0.2,
    num_predict=500,  # Équivalent de max_tokens
)
# llm = ChatGroq(
#     groq_api_key=gr_TOKEN,
#     model_name="llama-3.1-8b-instant",
#     temperature=0.2,
#     max_tokens=500,
# )
def run_rag(query, vectordb, llm):
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(query)

    # ✅ conversion Document → str
    contexts = [
        d.page_content if hasattr(d, "page_content") else d
        for d in docs
    ]

    prompt = f"""
Contexte:
{' '.join(contexts)}

Question:
{query}

Réponse:
"""
    answer = llm.invoke(prompt).content
    return answer, contexts


# llm = ChatGoogleGenerativeAI(
#    model="gemini-2.0-flash",
#     api_key=Gemini_API_KEY,
#     temperature=0.2, 
# )
@app.route('/', methods=['GET'])
def index():
    # On passe l'historique à la page HTML
    return render_template('index.html', chat_history=CHAT_HISTORY)

@app.route('/reset', methods=['POST'])
def reset_chat():
    global CHAT_HISTORY
    CHAT_HISTORY = [] # Vider l'historique
    flash("Discussion effacée !", 'success')
    return redirect(url_for('index'))

@app.route('/process', methods=['POST'])
def process_files():
    if 'pdf_files' not in request.files:
        return redirect(url_for('index'))
    
    files = request.files.getlist('pdf_files')
    docs = []
    
    try:
        for file in files:
            if file.filename and file.filename.endswith('.pdf'):
                filename = secure_filename(file.filename)
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)
                loader = PyPDFLoader(filepath)
                docs.extend(loader.load())
        
        if docs:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_documents(docs)
            vectordb = FAISS.from_documents(chunks, embedder)
            with open(VECTORSTORE_PATH, "wb") as f:
                pickle.dump(vectordb, f)
            flash(f"Succès ! {len(docs)} pages indexées.", 'success')
        else:
            flash("Aucun PDF valide.", 'error')
            
    except Exception as e:
        flash(f"Erreur: {str(e)}", 'error')

    return redirect(url_for('index'))

@app.route('/ask', methods=['POST'])
def ask_question():
    global CHAT_HISTORY
    query = request.form.get('query')
    
    if not query:
        return redirect(url_for('index'))

    # 1. Ajouter la question de l'utilisateur à l'historique
    CHAT_HISTORY.append({"role": "user", "content": query})

    if not os.path.exists(VECTORSTORE_PATH):
        CHAT_HISTORY.append({"role": "ai", "content": "Veuillez d'abord uploader des documents. et Tu dois aussi répondre de manière naturelle si l'utilisateur fait une interaction humaine, comme dire (hello), poser une question informelle ou demander de l'aide.", "sources": []})
        return redirect(url_for('index'))

    with open(VECTORSTORE_PATH, "rb") as f:
        vectordb = pickle.load(f)

    retriever = vectordb.as_retriever()
    relevant_docs = retriever.invoke(query)
    context = "\n\n".join(d.page_content for d in relevant_docs)

    # prompt = f"""Tu es un assistant IA utile. Utilise le contexte ci-dessous pour répondre.
    # Contexte: {context}
    # Question: {query}
    # Réponse:"""
    prompt = f"""
Tu es un assistant expert en enseignement et révision de cours. 
Ton rôle est de répondre aux questions des utilisateurs en utilisant UNIQUEMENT le contenu des documents fournis si le contenu n'existe pas repond avec non cela est pas mentionne dans le contenu. 
Tu dois aussi répondre de manière naturelle si l'utilisateur fait une interaction humaine, seulement poser une question informelle ou demander de l'aide n'ajoute rien.
tu dois repondre avec la langue utilisée par l'utilisateur dans sa question.

IMPORTANT - FORMAT DE RÉPONSE :
- Fournis la réponse directement en HTML valide(ex  <p>, <div>, <ol>, <li> ,<ul>, <h1>, <h2>, <h3> ...), sans balises <html> ou <body>.
- Ne jamais utiliser de blocs de code (```).
- Toutes tes réponses doivent être en HTML valide, prêt à être affiché sur une page web.
Règles de réponse :
- Si la question concerne le contenu des documents, répond de façon détaillée ou sous forme de résumé selon le contexte.
- Important: Toujours utiliser le contexte fourni pour répondre aux questions liées aux documents.

Contexte:
{context}

Question de l'utilisateur :
{query}

Réponse HTML :
"""

    try:
        response = llm.invoke(prompt)
        answer = response.content
        sources = [d.page_content[:100] + "..." for d in relevant_docs]
        
        # 2. Ajouter la réponse de l'IA à l'historique
        CHAT_HISTORY.append({"role": "ai", "content": answer, "sources": sources})
        
    except Exception as e:
        CHAT_HISTORY.append({"role": "ai", "content": f"Erreur: {str(e)}", "sources": []})

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)