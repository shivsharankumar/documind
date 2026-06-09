from contextlib import asynccontextmanager

from fastapi import FastAPI
from google import genai
from groq import AsyncGroq

from documind.agent.react_agent import ReActAgent
from documind.api.routes import agent as agent_route
from src.documind.api.routes import (
    ask,  # new
    chat,
)
from src.documind.config import settings
from src.documind.rag import RAGPipeline
from src.documind.retriever import Retriever


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create the Groq client once and reuse it
    app.state.groq = AsyncGroq(api_key=settings.groq_api_key)
    app.state.gemini = genai.Client(api_key=settings.gemini_api_key)
    retriever = Retriever()
    await retriever.setup()
    rag = RAGPipeline(retriever)
    app.state.rag = rag
    # Agent gets access to both retriever and RAG pipeline
    app.state.agent = ReActAgent(retriever=retriever, rag_pipeline=rag)
    yield
    # Shutdown: close the connection
    await app.state.groq.close()


app = FastAPI(
    title="DocuMind",
    version="0.1.0",
    lifespan=lifespan,
)


app.include_router(chat.router, prefix="/v1")
app.include_router(ask.router, prefix="/v1")
app.include_router(agent_route.router, prefix="/v1")


# A simple health check — useful for "is the server alive?"
@app.get("/health")
async def health():
    return {"status": "ok"}
