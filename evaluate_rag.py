"""
Script simple pour tester votre système RAG avec DeepEval + Ollama
Import du LLM et vectorstore depuis app.py - Aucune duplication!
"""

import os
import pickle
from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM
from langchain_ollama import OllamaLLM

# IMPORT DEPUIS VOTRE APP.PY
from app import llm, embedder, VECTORSTORE_PATH, run_rag
from questions_evaluation import questions

# ============================================
# CONFIGURATION
# ============================================

OLLAMA_MODEL = "gpt-oss:120b-cloud"  # Votre modèle Ollama


# ============================================
# WRAPPER OLLAMA POUR DEEPEVAL
# ============================================

class OllamaEvaluator(DeepEvalBaseLLM):
    """Wrapper pour utiliser Ollama avec DeepEval"""
    
    def __init__(self, model_name="gpt-oss:120b-cloud"):
        self.model = OllamaLLM(model=model_name)
        self.model_name = model_name
        
    def load_model(self):
        return self.model
    
    def generate(self, prompt: str) -> str:
        try:
            return self.model.invoke(prompt)
        except Exception as e:
            print(f" Erreur Ollama: {e}")
            return "Erreur de génération"
    
    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)
    
    def get_model_name(self) -> str:
        return f"Ollama-{self.model_name}"


# ============================================
# FONCTIONS UTILITAIRES
# ============================================

def load_vectorstore():
    """Charge le vectorstore FAISS depuis votre app"""
    if not os.path.exists(VECTORSTORE_PATH):
        print(f" Fichier {VECTORSTORE_PATH} introuvable!")
        print(" Uploadez d'abord des PDFs via l'application Flask")
        return None
    
    with open(VECTORSTORE_PATH, "rb") as f:
        return pickle.load(f)


def get_contexts(query, vectordb, k=5):
    """Récupère les contextes pertinents"""
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)
    return [d.page_content for d in docs]


def generate_answer(query, vectordb):
    """
    Génère une réponse en utilisant la fonction run_rag de app.py
    Cela garantit que vous testez exactement le même système que votre app!
    """
    answer, contexts = run_rag(query, vectordb, llm)
    return answer, contexts


# ============================================
# TESTS AUTOMATIQUES
# ============================================

def run_evaluation():
    """Lance l'évaluation complète"""
    
    print("\n" + "="*70)
    print(" ÉVALUATION RAG")
    print("="*70 + "\n")
    
    # 1. Charger le vectorstore
    vectordb = load_vectorstore()
    if not vectordb:
        return
    
    # 2. Initialiser Ollama
    try:
        ollama_eval = OllamaEvaluator(model_name=OLLAMA_MODEL)
        ollama_eval.generate("Test")
    except Exception as e:
        print(f" Erreur Ollama: {e}")
        return
    
    # 3. Charger les questions depuis le fichier
    print(f" {len(questions)} questions à évaluer\n")
    
    # 4. Créer les test cases
    test_cases = []
    
    for i, query in enumerate(questions, 1):
        try:
            actual, contexts = generate_answer(query, vectordb)
            test_case = LLMTestCase(
                input=query,
                actual_output=actual,
                retrieval_context=contexts
            )
            test_cases.append(test_case)
        except Exception as e:
            print(f" Erreur question {i}: {e}")
    
    # 5. Configurer les métriques
    metrics = [
        AnswerRelevancyMetric(threshold=0.5, model=ollama_eval, include_reason=False),
        FaithfulnessMetric(threshold=0.5, model=ollama_eval, include_reason=False),
    ]
    
    # 6. Évaluer
    print(" Évaluation en cours...\n")
    
    try:
        results = evaluate(test_cases=test_cases, metrics=metrics)
        
        # 7. Afficher les résultats de façon simple
        print("\n" + "="*70)
        print(" RÉSULTATS")
        print("="*70 + "\n")
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"{'='*70}")
            print(f"Question {i}: {test_case.input}")
            print(f"{'='*70}")
            print(f"\n Réponse:\n{test_case.actual_output}\n")
            
            # Récupérer les scores pour ce test case
            relevancy_score = None
            faithfulness_score = None
            
            if hasattr(results, 'test_results'):
                for result in results.test_results:
                    if result.input == test_case.input:
                        for metric_result in result.metrics_data:
                            if 'Answer Relevancy' in metric_result.name:
                                relevancy_score = metric_result.score
                            elif 'Faithfulness' in metric_result.name:
                                faithfulness_score = metric_result.score
            
            print(f" Métriques:")
            if relevancy_score is not None:
                print(f"   • Answer Relevancy: {relevancy_score:.2f}")
            if faithfulness_score is not None:
                print(f"   • Faithfulness: {faithfulness_score:.2f}")
            print()
        
        # Score global
        print("="*70)
        print(" SCORE GLOBAL")
        print("="*70)
        if hasattr(results, 'test_results') and len(results.test_results) > 0:
            all_relevancy = []
            all_faithfulness = []
            
            for result in results.test_results:
                for metric_result in result.metrics_data:
                    if 'Answer Relevancy' in metric_result.name:
                        all_relevancy.append(metric_result.score)
                    elif 'Faithfulness' in metric_result.name:
                        all_faithfulness.append(metric_result.score)
            
            if all_relevancy:
                avg_relevancy = sum(all_relevancy) / len(all_relevancy)
                print(f"Answer Relevancy moyen: {avg_relevancy:.2f}")
            if all_faithfulness:
                avg_faithfulness = sum(all_faithfulness) / len(all_faithfulness)
                print(f"Faithfulness moyen: {avg_faithfulness:.2f}")
        
        print("\n Évaluation terminée!\n")
        
        return results
        
    except Exception as e:
        print(f"\n Erreur: {e}")


# ============================================
# TEST RAPIDE D'UNE SEULE QUESTION
# ============================================

def test_single_question(query):
    """Test rapide d'une seule question"""
    
    print("\n" + "="*70)
    print(f"Question: {query}")
    print("="*70 + "\n")
    
    vectordb = load_vectorstore()
    if not vectordb:
        return
    
    actual_output, contexts = generate_answer(query, vectordb)
    
    print(f" Réponse:\n{actual_output}\n")
    
    # Évaluation optionnelle
    if input(" Évaluer avec DeepEval? (o/n): ").lower() == 'o':
        print("\n Évaluation...\n")
        
        ollama_eval = OllamaEvaluator(model_name=OLLAMA_MODEL)
        test_case = LLMTestCase(
            input=query,
            actual_output=actual_output,
            retrieval_context=contexts
        )
        
        metrics = [
            AnswerRelevancyMetric(threshold=0.5, model=ollama_eval, include_reason=False),
            FaithfulnessMetric(threshold=0.5, model=ollama_eval, include_reason=False),
        ]
        
        results = evaluate(test_cases=[test_case], metrics=metrics)
        
        # Afficher les scores
        print("\n Métriques:")
        if hasattr(results, 'test_results') and len(results.test_results) > 0:
            for metric_result in results.test_results[0].metrics_data:
                print(f"   • {metric_result.name}: {metric_result.score:.2f}")


# ============================================
# TEST COMPARATIF (BONUS)
# ============================================

def compare_rag_configs():
    """Compare différentes configurations de votre RAG"""
    
    print("\n" + "="*70)
    print(" TEST COMPARATIF - Différentes configurations K")
    print("="*70 + "\n")
    
    vectordb = load_vectorstore()
    if not vectordb:
        return
    
    query = input(" Entrez une question de test: ").strip()
    if not query:
        print(" Question vide!")
        return
    
    k_values = [3, 5, 10]
    
    print(f"\n Comparaison pour la question: {query}\n")
    
    for k in k_values:
        print(f"\n{'='*70}")
        print(f" Test avec K={k} contextes")
        print('='*70)
        
        # Récupérer avec différents K
        retriever = vectordb.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(query)
        contexts = [d.page_content for d in docs]
        
        # Générer réponse
        context_text = "\n\n".join(contexts)
        prompt = f"""Tu es un assistant expert. Réponds à la question en utilisant UNIQUEMENT le contexte fourni.

Contexte:
{context_text}

Question:
{query}

Réponse:"""
        
        response = llm.invoke(prompt)
        answer = response.content
        
        print(f"\n Réponse (K={k}):")
        print(answer)
        print(f"\n Longueur: {len(answer)} caractères")
        print(f" Contextes utilisés: {k}")


# ============================================
# MENU PRINCIPAL
# ============================================

def main():
    """Menu principal"""
    
    print("\n" + "="*70)
    print(" ÉVALUATION RAG")
    print("="*70)
    print("\n1. Évaluation complète")
    print("2. Test rapide")
    print("3. Quitter")
    
    choice = input("\nChoix (1/2/3): ").strip()
    
    if choice == "1":
        run_evaluation()
    elif choice == "2":
        query = input("\n Question: ").strip()
        if query:
            test_single_question(query)
    elif choice == "3":
        print("\n Au revoir!")
    else:
        print("\n Choix invalide!")


# ============================================
# POINT D'ENTRÉE
# ============================================

if __name__ == "__main__":
    main()