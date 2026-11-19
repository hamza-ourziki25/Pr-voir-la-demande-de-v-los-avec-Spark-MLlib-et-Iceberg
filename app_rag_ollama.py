import hashlib
from pathlib import Path

import chromadb
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

UPLOAD_DIR = Path("./uploads_rag")
UPLOAD_DIR.mkdir(exist_ok=True)
CHROMA_DIR = "./chroma_db_rag"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:1b"

PROMPT_TEMPLATE = """Tu es un assistant qui répond à des questions sur un document PDF.
Réponds uniquement à partir du CONTEXTE ci-dessous.
Si la réponse ne se trouve pas dans le CONTEXTE, réponds exactement : JE NE SAIS PAS.
N'invente rien. Réponds en français, de façon claire et concise.

CONTEXTE :
{context}

QUESTION : {question}

RÉPONSE :"""


# ------------------------------------------------------- Partie traitement
def rechercher_passages(question, nom_collection, embeddings, client, k=2):
    """Étape 5 : return the k chunks closest to the question."""
    collection = client.get_collection(nom_collection)
    vecteur_question = embeddings.embed_query(question)
    resultat = collection.query(
        query_embeddings=[vecteur_question],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    passages = []
    for texte, meta, dist in zip(
        resultat["documents"][0],
        resultat["metadatas"][0],
        resultat["distances"][0],
    ):
        passages.append({
            "texte": texte,
            "page": meta.get("page"),
            "source": meta.get("source"),
            "distance": dist,
        })
    return passages


def construire_prompt(question, passages):
    """Étape 6 : join the passages into a context and fill the template."""
    contexte = "\n\n".join(f"[Page {p['page']}]\n{p['texte']}" for p in passages)
    return PROMPT_TEMPLATE.format(context=contexte, question=question)


def generer_reponse(question, passages, llm):
    """Étape 7 : build the prompt, ask Llama, return (answer, prompt)."""
    prompt = construire_prompt(question, passages)
    reponse = llm.invoke(prompt)
    return reponse.content, prompt


# ---------------------------------------------------------------- Interface
st.set_page_config(page_title="Assistant RAG local", layout="wide")
st.title("Assistant documentaire RAG — Ollama")
st.write("Téléversez un PDF, indexez-le, puis posez votre question.")

valeurs_initiales = {
    "selection_pdf": None,
    "collection_active": None,
    "document_actif": None,
    "nombre_chunks": 0,
    "dernier_resultat": None,
}
for cle, valeur in valeurs_initiales.items():
    if cle not in st.session_state:
        st.session_state[cle] = valeur

with st.sidebar:
    st.header("1. Charger un document")
    # Étape 1 : récupérer le PDF depuis Streamlit
    fichier = st.file_uploader("Choisir un PDF", type=["pdf"])

    # Activité 9 : un autre PDF sélectionné = ancien état réinitialisé
    selection = (
        hashlib.sha256(fichier.getvalue()).hexdigest() if fichier else None
    )
    if selection != st.session_state.selection_pdf:
        st.session_state.selection_pdf = selection
        st.session_state.collection_active = None
        st.session_state.document_actif = None
        st.session_state.nombre_chunks = 0
        st.session_state.dernier_resultat = None

    if st.button("Indexer le PDF", disabled=fichier is None):
        try:
            with st.spinner("Indexation du document..."):
                empreinte = hashlib.sha256(fichier.getvalue()).hexdigest()
                chemin_pdf = UPLOAD_DIR / f"{empreinte}.pdf"
                if not chemin_pdf.exists():
                    chemin_pdf.write_bytes(fichier.getvalue())

                # Étape 2 : charger le texte du PDF
                pages = PyPDFLoader(str(chemin_pdf)).load()
                for page in pages:
                    page.metadata["source"] = fichier.name

                # Étape 3 : découper en chunks
                splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                    encoding_name="cl100k_base",
                    chunk_size=768,
                    chunk_overlap=16, # keep the conection between 2 shunks
                )
                chunks = splitter.split_documents(pages)
                chunks = [c for c in chunks if c.page_content.strip()]
                if not chunks:
                    raise ValueError(
                        "Aucun texte extrait : PDF scanné ? (OCR nécessaire)"
                    )

                # Étape 4 : embeddings + stockage dans ChromaDB
                embeddings = OllamaEmbeddings(model=EMBED_MODEL)
                client = chromadb.PersistentClient(path=CHROMA_DIR)
                nom_collection = f"pdf_nomic_{empreinte[:24]}"
                collection = client.get_or_create_collection(name=nom_collection)

                ids = [f"{empreinte[:16]}_{i}" for i in range(len(chunks))]
                deja = set(collection.get(include=[])["ids"])
                a_faire = [i for i in range(len(chunks)) if ids[i] not in deja]

                TAILLE_LOT = 64
                for debut in range(0, len(a_faire), TAILLE_LOT):
                    indices = a_faire[debut:debut + TAILLE_LOT]
                    lot = [chunks[i] for i in indices]
                    textes = [c.page_content for c in lot]
                    collection.upsert(
                        ids=[ids[i] for i in indices],
                        documents=textes,
                        embeddings=embeddings.embed_documents(textes),
                        metadatas=[
                            {
                                "source": c.metadata.get("source", fichier.name),
                                "page": int(c.metadata.get("page", 0)) + 1,
                            }
                            for c in lot
                        ],
                    )

            st.session_state.collection_active = nom_collection
            st.session_state.document_actif = fichier.name
            st.session_state.nombre_chunks = collection.count()
            st.write(f"Taille : {len(fichier.getvalue())/1024:.0f} Ko · Pages : {len(pages)} · Chunks : {len(chunks)}")
            st.session_state.dernier_resultat = None
            st.success(
                "Indexation terminée !" if a_faire
                else "PDF déjà indexé : collection réutilisée."
            )
        except Exception as exc:
            st.error(f"Échec de l'indexation : {exc}")

    if st.session_state.collection_active:
        st.info(
            f"PDF actif : {st.session_state.document_actif}\n\n"
            f"Chunks indexés : {st.session_state.nombre_chunks}"
        )

# ---- Zone principale : question
st.header("Poser une question")
with st.form("formulaire_question"):
    question = st.text_input("Votre question sur le PDF actif")
    demander = st.form_submit_button(
        "Obtenir une réponse",
        disabled=st.session_state.collection_active is None,
    )

# ---- Calcul : on ne fait que remplir st.session_state.dernier_resultat
if demander:
    if not question.strip():
        st.warning("Saisissez d'abord une question.")
    else:
        try:
            with st.spinner("Recherche et génération avec Llama 3.2..."):
                embeddings = OllamaEmbeddings(model=EMBED_MODEL)
                client = chromadb.PersistentClient(path=CHROMA_DIR)
                llm = ChatOllama(model=LLM_MODEL, temperature=0)

                passages = rechercher_passages(
                    question, st.session_state.collection_active, embeddings, client
                )
                if passages:
                    reponse, prompt = generer_reponse(question, passages, llm)
                    st.session_state.dernier_resultat = {
                        "question": question,
                        "reponse": reponse,
                        "passages": passages,
                        "prompt": prompt,
                    }
                else:
                    st.warning("Aucun passage trouvé dans la collection.")
        except Exception as exc:
            st.error(f"Erreur lors du RAG : {exc}")

# ---- Étape 8 : affichage (hors du `if demander`, donc persistant)
resultat = st.session_state.dernier_resultat
if resultat:
    st.subheader("Réponse de Llama 3.2")
    st.caption(f"Question : {resultat['question']}")
    st.markdown(resultat["reponse"])

    with st.expander("Voir les passages récupérés"):
        for i, p in enumerate(resultat["passages"], 1):
            st.markdown(f"**Passage {i} — {p['source']} — Page {p['page']}**")
            st.write(p["texte"])

    with st.expander("Voir le prompt augmenté"):
        st.code(resultat["prompt"], language="text")