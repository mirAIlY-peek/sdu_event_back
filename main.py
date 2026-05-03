import json
import os
import shutil
from datetime import datetime
from threading import Thread

from dotenv import load_dotenv

load_dotenv()


import firebase_admin
from firebase_admin import credentials, firestore as fs
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


app = FastAPI(title="UniBuddy AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SYNC_API_KEY = os.getenv("SYNC_API_KEY")
MODEL_NAME = "google/gemini-2.5-flash-lite"
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
RAG_JSON_PATH = os.getenv("RAG_DATASET_PATH", "rag_dataset.json")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY is not set in .env")

if not SYNC_API_KEY:
    raise ValueError("SYNC_API_KEY is not set in .env")


_firestore_client = None


def _ensure_firebase_app():
    """Инициализирует firebase-admin один раз; путь к JSON — из .env."""
    global _firestore_client
    if _firestore_client is not None:
        return _firestore_client

    path = os.getenv("FIREBASE_CREDENTIALS_PATH") or os.getenv(
        "GOOGLE_APPLICATION_CREDENTIALS"
    )
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(
            "Укажите FIREBASE_CREDENTIALS_PATH или GOOGLE_APPLICATION_CREDENTIALS "
            "в .env (файл ключа сервис-аккаунта Firebase)."
        )

    cred = credentials.Certificate(path)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    _firestore_client = fs.client()
    return _firestore_client


def fetch_events_from_firestore() -> list[Document]:
    db = _ensure_firebase_app()
    out: list[Document] = []
    for snap in db.collection("events").stream():
        data = snap.to_dict() or {}
        
        # Берем данные
        title = str(data.get("title") or data.get("name") or "Без названия")
        desc = str(data.get("description") or "Описание отсутствует")
        location = str(data.get("location") or "Место не указано")
        date = str(data.get("date") or "Дата не указана")
        category = str(data.get("category") or "Общее")
        
        
        page_content = (
            f"МЕРОПРИЯТИЕ: {title}\n"
            f"Краткая суть: {title}\n"
            f"О ЧЕМ ЭТО: {desc}\n"
            f"ГДЕ И КОГДА: Пройдет {date} в {location}.\n"
            f"КАТЕГОРИЯ: {category}."
        )
        
        meta = {
            "id": snap.id,
            "type": "event",
            "title": title,
            "name": title,
            "category": category,
            "date": date,
        }
        out.append(Document(page_content=page_content, metadata=meta))
    return out


def _load_faq_documents() -> list[Document]:
    with open(RAG_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        Document(page_content=item["page_content"], metadata=item["metadata"])
        for item in data
    ]


def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        model=MODEL_NAME,
        temperature=0.4,
        default_headers={
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "UniBuddy AI",
        },
    )


def _build_rag_prompt():
    system_prompt = (
        "Ты — UniBuddy SDU. Помогаешь студентам.\n\n"
        "ЯЗЫК:\n"
        "- Отвечай строго на языке пользователя (RU или KZ)\n\n"
        "КОНТЕКСТ:\n"
        "{context}\n\n"
        "ПРАВИЛА:\n"
        "- Кратко (2-3 абзаца)\n"
        "- Если есть несколько вариантов — перечисли все\n"
        "- Названия выделяй жирным\n"
        "- Если информации нет — скажи: 'У меня пока нет данных об этом'\n"
    )
    return ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )


def _make_rag_chain(vectorstore: Chroma, llm: ChatOpenAI):
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.7},
    )
    question_chain = create_stuff_documents_chain(llm, _build_rag_prompt())
    return create_retrieval_chain(retriever, question_chain)


def _update_or_create_vectorstore(
    documents: list[Document], embeddings: OpenAIEmbeddings, existing_vectorstore: Chroma = None
) -> Chroma:
    """Безопасно обновляет или создает ChromaDB, избегая блокировок файлов в Windows."""
    if existing_vectorstore is not None:
        print("⏳ Очистка существующих записей в Chroma...")
        
        curr_data = existing_vectorstore.get()
        ids = curr_data['ids']
        if ids:
            existing_vectorstore.delete(ids=ids)
        
        # Добавляем новые документы в существующий объект
        print(f"⏳ Добавление {len(documents)} новых документов в существующую базу...")
        existing_vectorstore.add_documents(documents)
        return existing_vectorstore
    else:
        
        if os.path.exists(CHROMA_DIR):
            try:
                shutil.rmtree(CHROMA_DIR)
            except PermissionError:
                print("⚠️ Не удалось удалить папку (заблокирована). Используем существующую.")
                return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

        return Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=CHROMA_DIR,
        )



def initialize_ai():
    try:
        print("⏳ Loading embeddings...")
        embeddings = OpenAIEmbeddings(
            api_key=OPENROUTER_API_KEY,
            openai_api_base="https://openrouter.ai/api/v1",
            model="openai/text-embedding-3-small",
        )

        print("⏳ Loading LLM...")
        llm = _build_llm()

        print("⏳ Creating vector DB...")
        documents = _load_faq_documents()
        vectorstore = _update_or_create_vectorstore(documents, embeddings)

        app.state.embeddings = embeddings
        app.state.llm = llm
        app.state.vectorstore = vectorstore
        app.state.rag_chain = _make_rag_chain(vectorstore, llm)

        print("🔥 UniBuddy AI Ready!")

    except Exception as e:
        print(f"❌ Startup Error: {e}", flush=True)
        import traceback; traceback.print_exc()


@app.on_event("startup")
async def startup_event():
    app.state.embeddings = None
    app.state.llm = None
    app.state.vectorstore = None
    app.state.rag_chain = None

    Thread(target=initialize_ai, daemon=True).start()

    print("🚀 FastAPI server started")



def _verify_sync_key(x_api_key: str | None) -> None:
    if x_api_key != SYNC_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.post("/sync")
def sync_rag(x_api_key: str | None = Header(default=None)):
    """
    Безопасная пересборка Chroma без удаления папки (для Windows).
    """
    _verify_sync_key(x_api_key)

    embeddings: OpenAIEmbeddings = app.state.embeddings
    llm: ChatOpenAI = app.state.llm
    current_vectorstore: Chroma = app.state.vectorstore # Берем текущий объект

    print("⏳ Синхронизация: загрузка документов...")
    faq_docs = _load_faq_documents()
    try:
        event_docs = fetch_events_from_firestore()
    except Exception as e:
        raise HTTPException(
            status_code=502, detail=f"Firestore error: {e!s}"
        ) from e

    all_docs = faq_docs + event_docs
    
    
    new_vectorstore = _update_or_create_vectorstore(all_docs, embeddings, current_vectorstore)
    
    app.state.vectorstore = new_vectorstore
    app.state.rag_chain = _make_rag_chain(new_vectorstore, llm)

    return {
        "status": "ok",
        "events_synced": len(event_docs),
        "total_chunks": len(all_docs),
    }



class QueryRequest(BaseModel):
    query: str


class RecommendRequest(BaseModel):
    interests: list[str] = Field(default_factory=list)
    user_name: str | None = None



@app.post("/ask")
async def ask_unibuddy(request: QueryRequest):
    rag_chain = app.state.rag_chain
    if rag_chain is None:
        return {"error": "System still loading"}

    response = rag_chain.invoke({"input": request.query})

    unique_sources = []
    seen = set()
    for doc in response["context"]:
        name = (
            doc.metadata.get("title")
            or doc.metadata.get("name")
            or "Unknown"
        )
        if name not in seen:
            seen.add(name)
            unique_sources.append(doc.metadata)

    return {
        "question": request.query,
        "answer": response["answer"],
        "sources": unique_sources,
    }

@app.get("/")
async def root():
    return {
        "status": "online",
        "ai_ready": app.state.rag_chain is not None,
    }

@app.post("/recommend")
async def recommend(request: RecommendRequest):
    vectorstore: Chroma | None = app.state.vectorstore
    llm: ChatOpenAI | None = app.state.llm
    if vectorstore is None or llm is None:
        raise HTTPException(status_code=503, detail="Vector store not ready")

    interests_label = (
        ", ".join(request.interests)
        if request.interests
        else "университетские мероприятия и клубы"
    )
    query = f"События и мероприятия для интересов: {interests_label}"

    
    try:
        results = vectorstore.similarity_search(
            query,
            k=5,
            filter={"type": "event"},
        )
    except Exception:
        
        raw = vectorstore.similarity_search(query, k=24)
        results = [d for d in raw if d.metadata.get("type") == "event"][:5]

    if len(results) < 3 and request.interests:
        raw = vectorstore.similarity_search(query, k=24)
        merged = [d for d in raw if d.metadata.get("type") == "event"]
        seen_ids = {d.metadata.get("id") for d in results}
        for d in merged:
            if d.metadata.get("id") not in seen_ids:
                results.append(d)
                seen_ids.add(d.metadata.get("id"))
            if len(results) >= 5:
                break

    results = results[:5]
    events_text = "\n".join(
        f"- {d.metadata.get('title', 'Ивент')}: {d.page_content[:400]}"
        for d in results
    )

    rec_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Ты — UniBuddy. Студент {name} интересуется: {interests}.\n"
                "По списку ивентов напиши 1–2 коротких предложения: почему эти "
                "мероприятия ему подходят. Только связный текст, без маркированного списка. "
                "Язык ответа — русский.",
            ),
            ("human", "Ивенты:\n{events}"),
        ]
    )
    chain = rec_prompt | llm
    name = request.user_name or "студент"
    msg = chain.invoke(
        {
            "name": name,
            "interests": interests_label,
            "events": events_text or "(ничего не найдено)",
        }
    )
    explanation = getattr(msg, "content", str(msg))

    recommendations = [
        {
            "id": d.metadata.get("id"),
            "title": d.metadata.get("title") or d.metadata.get("name"),
            "category": d.metadata.get("category"),
            "date": d.metadata.get("date"),
        }
        for d in results
    ]

    return {
        "interests": request.interests,
        "recommendations": recommendations,
        "explanation": explanation,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
