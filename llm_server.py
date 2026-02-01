from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
import os
import json
from contextlib import asynccontextmanager
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

from llm_agent.pipeline import CompleteMealPlannerPipeline
from llm_agent.utils.logger import logger
# ============================= 1. CONFIGURATION =============================
# Load environment variables from .env file
load_dotenv()

# --- Configuration (Environment Variables) ---
MONGODB_URI = os.environ.get("MONGODB_URI")
DATABASE_NAME = os.environ.get("DATABASE_NAME")
FOOD_COLLECTION = os.environ.get("FOOD_COLLECTION")
MANUAL_COLLECTION = os.environ.get("MANUAL_COLLECTION")
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL")
VLLM_MODEL_NAME = os.environ.get("VLLM_MODEL_NAME")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY")
BACKEND_URL = os.environ.get("BACKEND_URL")
VLLM_TEMPERATURE = float(os.environ.get("VLLM_TEMPERATURE", 0.7))
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME")

# --- Global Pipeline Instance ---
pipeline: CompleteMealPlannerPipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager for the FastAPI application.
    Initializes the AI pipeline on startup and closes it on shutdown.
    """
    global pipeline
    try:
        if not VLLM_API_KEY:
             logger.warning("⚠️  WARNING: VLLM_API_KEY is not set. LLM features may fail.")

        pipeline = CompleteMealPlannerPipeline(
            mongodb_uri=MONGODB_URI,
            database_name=DATABASE_NAME,
            food_collection=FOOD_COLLECTION,
            manual_collection=MANUAL_COLLECTION,
            vllm_base_url=VLLM_BASE_URL,
            vllm_model_name=VLLM_MODEL_NAME,
            vllm_api_key=VLLM_API_KEY,
            vllm_temperature=VLLM_TEMPERATURE,
            backend_url=BACKEND_URL,
            embedding_model_name=EMBEDDING_MODEL_NAME,
        )
        logger.info("✅ AI Pipeline Initialized Successfully (from env)")
    except Exception as e:
        logger.error(f"❌ Failed to initialize AI Pipeline: {e}")

    yield

    if pipeline:
        pipeline.close()
        logger.info("✅ AI Pipeline Closed")

app = FastAPI(
    title="NutriPlan LLM Server",
    description="""
    ## Overview
    This API serves as the AI backend for the NutriPlan application.
    It provides endpoints for:
    - **Conversational AI**: Context-aware chat with RAG support for meal planning.
    - **Food Search**: Vector-based semantic search for foods.
    - **Manual Search**: Retrieval-augmented search over the user manual.

    ## Features
    - **Streaming Responses**: The `/chat` endpoint uses Server-Sent Events (SSE) for real-time tokens.
    - **RAG Integration**: Automatically retrieves context from MongoDB vector stores.
    - **Tool Use**: Capable of executing tools for web search and backend data retrieval.
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# --- CORS ---
allowed_origins_env = os.environ.get("ALLOWED_ORIGINS", "")
if allowed_origins_env:
    origins = [origin.strip() for origin in allowed_origins_env.split(",")]
else:
    # Default fallback
    origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Exception Handlers ---
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"❌ Validation Error: {exc.errors()}")
    try:
        body = await request.json()
        logger.error(f"❌ Request Body: {json.dumps(body, indent=2)}")
    except:
        pass
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )

# ============================= 2. DATA MODELS =============================
class ChatRequest(BaseModel):
    # Frontend sends a format similar to OpenAI chat completion
    messages: List[Dict[str, str]] = Field(
        ...,
        description="List of messages in the conversation (system, user, assistant).",
        json_schema_extra={
            "example": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"}
            ]
        }
    )
    # TODO: Currently not used, but can be integrated in future
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = True

class ChatResponse(BaseModel):
    # Since we stream, this model is mainly for documentation
    status: str = Field(..., description="Status of the chunk (token, thinking, done)")
    message: Optional[str] = Field(None, description="Thinking process message")
    content: Optional[str] = Field(None, description="The text content token of the response")
    commands: Optional[List[Dict]] = Field(None, description="Frontend commands extracted from the response")

class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        description="The search query text.",
        json_schema_extra={"example": "high protein breakfast"}
    )
    k: int = Field(5, description="Number of results to return.", ge=1, le=20)
    filters: Optional[Dict[str, Any]] = Field(
        None,
        description="MongoDB style filters for advanced search.",
        json_schema_extra={"example": {"nutrition.calories": {"$lt": 500}}}
    )

class DocumentMetadata(BaseModel):
    name: Optional[str] = Field(None, description="Name of the food item")
    score: Optional[float] = Field(None, description="Vector similarity score")
    categories: Optional[List[str]] = Field(None, description="Food categories")
    nutrition: Optional[Dict[str, Any]] = Field(None, description="Nutritional information")

class SearchResponse(BaseModel):
    data: List[Dict[str, Any]] = Field(..., description="List of search results including metadata and scores.")

# ============================= 3. API ENDPOINTS =============================

@app.get("/", tags=["Health"])
async def root():
    """
    **Health Check**

    Returns the status of the server to verify it is running.
    """
    return {"status": "ok", "message": "NutriPlan LLM Server is Running"}

@app.post(
    "/ai/chat",
    tags=["Chat"],
    summary="Chat with AI Agent",
    description="""
    Sends a message to the AI agent and receives a streaming response via Server-Sent Events (SSE).

    **Authorization**:
    - Optional `Authorization` header with `Bearer <token>` is required for personalized features (Meal Plan, Pantry).

    **Stream Format**:
    - The response is a stream of event data blocks: `data: <json>\n\n`.
    - JSON chunks can be:
      - `{"status": "thinking", "message": "..."}`: Agent is processing.
      - `{"status": "token", "content": "..."}`: Part of the answer text.
      - `{"status": "done", "commands": [...]}`: Stream complete, includes any UI commands.
    """,
    response_class=StreamingResponse
)
async def chat_endpoint(
    request: ChatRequest,
    authorization: Optional[str] = Header(None, description="Bearer token for authenticated user context")
):
    if not pipeline:
        raise HTTPException(status_code=503, detail="AI Pipeline not initialized")

    # Extract clean token
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split("Bearer ")[1]

    # Parse OpenAI-like messages
    messages = request.messages
    if not messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    # Last message is the user's current query
    last_message = messages[-1]
    # Verify if the last one is user
    if last_message["role"] != "user":
        raise HTTPException(status_code=400, detail="No user message")

    user_query = last_message["content"]

    # History is everything before the last message
    history = messages[:-1]

    async def event_generator():
        try:
             # The pipeline.chat returns a generator yielding dicts
            for chunk in pipeline.chat(user_query, history, token):
                yield f"{json.dumps(chunk)}\n"
        except Exception as e:
            # yield error object
            err_obj = {"status": "error", "message": str(e)}
            yield f"{json.dumps(err_obj)}\n"
            logger.error(f"Error in chat stream: {e}")

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")

@app.post(
    "/search/food",
    tags=["Search"],
    response_model=SearchResponse,
    summary="Semantic Food Search",
    description="Performs a vector-based semantic search on the food database."
)
async def search_food(request: SearchRequest):
    if not pipeline:
         raise HTTPException(status_code=503, detail="AI Pipeline not initialized")
    try:
        results = pipeline.search_food(request.query, request.k, request.filters)
        # Convert Documents to dicts
        return {"data": [doc.dict() for doc in results]}
    except Exception as e:
        logger.error(f"Search Food Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post(
    "/search/manual",
    tags=["Search"],
    response_model=SearchResponse,
    summary="User Manual Search",
    description="Performs a vector-based semantic search on the user manual / documentation."
)
async def search_manual(request: SearchRequest):
    if not pipeline:
         raise HTTPException(status_code=503, detail="AI Pipeline not initialized")
    try:
        results = pipeline.search_manual(request.query, request.k, request.filters)
        return {"data": results}
    except Exception as e:
         logger.error(f"Search Manual Error: {e}")
         raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("llm_server:app", host="0.0.0.0", port=8000, reload=True)
