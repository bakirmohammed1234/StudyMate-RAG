import os
import pickle
from flask import Flask, render_template, request, flash, redirect, url_for, session

# Tes imports LangChain
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
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
# Pour ton usage local, une liste globale suffit.
CHAT_HISTORY = [] 

print("Chargement du modèle d'embedding...")
embedder = HuggingFaceEmbeddings()

llm = ChatGroq(
    groq_api_key=gr_TOKEN,
    model_name="llama-3.1-8b-instant",
    temperature=0.2,
    max_tokens=500,
)
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
        CHAT_HISTORY.append({"role": "ai", "content": "Veuillez d'abord uploader des documents.", "sources": []})
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
    Tu es un assistant expert. Réponds à la question en utilisant UNIQUEMENT le contexte suivant.
    
    IMPORTANT - FORMAT DE RÉPONSE :
    Tu dois répondre directement en format HTML valide, sans balises <html> ou <body> et SANS blocs de code (```).
    
    Utilise ces balises pour structurer ta réponse :
    - <h3>Titre</h3> pour les titres de sections.
    - <strong>Mot important</strong> pour mettre en gras.
    - <ul><li>Point 1</li><li>Point 2</li></ul> pour les listes.
    - <p>Paragraphe</p> pour le texte normal.
    - <br> pour les sauts de ligne.
    
    Contexte :
    {context}

    Question :
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