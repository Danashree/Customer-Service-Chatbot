from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)

# Create Gemini LLM model (model name configurable; newer Google accounts no longer
# have access to gemini-2.5-flash, so .env can pin e.g. gemini-3.8-flash)
llm = ChatGoogleGenerativeAI(
    model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
    google_api_key=os.environ["GOOGLE_API_KEY"],
    temperature=0.1,
)
# Initialize instructor embeddings using the Hugging Face model
instructor_embeddings = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-large"
)
vectordb_file_path = "faiss_index" if os.path.exists("faiss_index") else os.path.join(os.path.dirname(__file__), "faiss_index")


def _get_dataset_path():
    candidates = [
        "../dataset/dataset.csv",
        "dataset/dataset.csv",
        os.path.join(os.path.dirname(__file__), "..", "dataset", "dataset.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return "../dataset/dataset.csv"


def get_active_faiss_path():
    """
    Locates and validates the currently active Task 1 FAISS version directory.
    Flow: active_version.json -> versions/<active_version>/ -> index.faiss & index.pkl.
    Fails safely without silent fallback if active_version.json is missing or corrupted.
    """
    import json
    candidates = [
        os.path.join(os.path.dirname(__file__), "versions"),
        "backend/versions",
        "versions",
    ]
    versions_dir = None
    for d in candidates:
        if os.path.isdir(d):
            versions_dir = d
            break

    if not versions_dir:
        raise FileNotFoundError(
            "Task 1 versions directory not found. Please initialize or run the pipeline first."
        )

    active_file = os.path.join(versions_dir, "active_version.json")
    if not os.path.exists(active_file):
        raise FileNotFoundError(
            f"Active version file '{active_file}' not found. Please activate a knowledgebase version."
        )

    try:
        with open(active_file, "r", encoding="utf-8") as f:
            active_data = json.load(f)
            active_version = active_data.get("active_version")
    except Exception as e:
        raise ValueError(f"Invalid active_version.json: {e}")

    if not active_version or not isinstance(active_version, str):
        raise ValueError(f"Invalid active version identifier: '{active_version}'")

    target_dir = os.path.join(versions_dir, active_version)
    has_faiss = os.path.exists(os.path.join(target_dir, "index.faiss"))
    has_pkl = os.path.exists(os.path.join(target_dir, "index.pkl"))

    if not (has_faiss and has_pkl):
        raise FileNotFoundError(
            f"Active FAISS version '{active_version}' is missing index files in '{target_dir}'."
        )

    return target_dir


def _get_faiss_path():
    """Returns the path to the currently active Task 1 FAISS index."""
    return get_active_faiss_path()


def create_vector_db():
    # Load data from FAQ sheet
    csv_file = _get_dataset_path()
    loader = CSVLoader(file_path=csv_file, source_column="prompt")
    data = loader.load()

    # Create a FAISS instance for vector database from 'data'
    vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)

    # Save vector database locally
    target_path = "faiss_index" if os.path.exists("faiss_index") and not os.path.isabs(vectordb_file_path) else os.path.join(os.path.dirname(__file__), "faiss_index")
    vectordb.save_local(target_path)


def get_qa_chain():
    # Load the vector database from the local folder
    target_path = _get_faiss_path()
    vectordb = FAISS.load_local(target_path, instructor_embeddings, allow_dangerous_deserialization=True)


    # Create a retriever for querying the vector database
    # No score_threshold: instructor embeddings are unnormalized, so the euclidean
    # relevance scale never reaches 0.7 and a threshold would empty the context.
    # Out-of-domain questions are still refused by the prompt's "I don't know" guard.
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    prompt_template = PromptTemplate(
        template="""Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}""",
        input_variables=["context", "question"]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # LCEL chain: retrieve -> format -> prompt -> LLM -> parse
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    class ChainWrapper:
        def __init__(self, chain):
            self._chain = chain

        def __call__(self, query):
            result = self._chain.invoke(query)
            return {"result": result}

    return ChainWrapper(chain)
