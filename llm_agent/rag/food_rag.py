from typing import List, Dict, Optional
import torch
from huggingface_hub import snapshot_download
from pymongo import MongoClient
from pymongo.errors import OperationFailure
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from llm_agent.utils.logger import logger
from llm_agent.utils.constants import CATEGORY_LABELS
from llm_agent.utils.messages import LogMessages

class FoodRAGPipeline:
    """RAG Pipeline for Food Database"""

    def __init__(
        self, mongodb_uri: str, database_name: str, collection_name: str, model_name: str
    ):
        logger.info(LogMessages.FOOD_RAG_INIT)

        # Embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Reranker model
        # Using a small, fast cross-encoder
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device="cuda" if torch.cuda.is_available() else "cpu")

        # MongoDB connection
        self.client = MongoClient(mongodb_uri)
        self.db = self.client[database_name]
        self.collection = self.db[collection_name]

        logger.info(LogMessages.FOOD_RAG_CONNECTED.format(db=database_name, collection=collection_name))
        logger.info(LogMessages.FOOD_RAG_COUNT.format(count=self.collection.count_documents({})))

    def search(self, query: str, k: int = 5, filters: Dict = None) -> List[Document]:
        """Hybrid Search: Vector Search with Pre-filtering"""

        # Case 1: Pure Filtering (No semantic query)
        if not query and filters:
            logger.info(LogMessages.FOOD_RAG_RESULT.format(query="[FILTER ONLY]", names=str(filters)))
            results = list(self.collection.find(filters).limit(k))

        # Case 2: Vector Search (with optional Pre-filtering)
        else:
            query_embedding = self.embeddings.embed_query(query)

            vector_search_stage = {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": k * 20,
                "limit": k,
            }

            # Inject pre-filter if exists
            if filters:
                vector_search_stage["filter"] = filters

            pipeline = [
                {"$vectorSearch": vector_search_stage},
                {
                    "$project": {
                        "name": 1,
                        "text_content": 1,
                        "categories": 1,
                        "nutrition": 1,
                        "property": 1,
                        "score": {"$meta": "vectorSearchScore"},
                    }
                }
            ]

            try:
                results = list(self.collection.aggregate(pipeline))
            except OperationFailure as e:
                # Fallback for missing index on filters
                if "needs to be indexed as filter" in str(e):
                    logger.warning(f"Vector search failed due to missing index: {e}. Falling back to standard DB search.")

                    fallback_filter = filters.copy() if filters else {}

                    # Improve fallback: If there involves a semantic query, try simple Regex match
                    if query and query.strip():
                        # Simple stopword removal to avoid "meals with egg" failing to match "Fried Egg"
                        stop_words = {"show", "me", "find", "suggest", "recommend", "meals", "meal", "with", "for", "a", "an", "the", "in", "on"}
                        words = query.strip().split()
                        filtered_words = [w for w in words if w.lower() not in stop_words]

                        if filtered_words:
                            # Join with | to match ANY of the keywords (OR logic) -> broader reach
                            # Or join with space? "Egg" -> "Egg"
                            # "Egg Salad" -> "Egg.*Salad" or "Egg Salad".
                            # Let's try matching the longest meaningful word or the sequence.
                            # Safest fallback: Match the sequence of remaining words.
                            safe_regex = " ".join(filtered_words)
                        else:
                            # If all were stop words, revert to original
                            safe_regex = query.strip()

                        fallback_filter["text_content"] = {"$regex": safe_regex, "$options": "i"}

                    results = list(self.collection.find(fallback_filter).limit(k))
                else:
                    raise e

        documents = []
        for result in results:
            nutrition = result.get("nutrition", {})
            nutrition_text = f"""Calories: {nutrition.get('calories', 'N/A')} kcal
Protein: {nutrition.get('proteins', 'N/A')}g
Carbs: {nutrition.get('carbs', 'N/A')}g
Fat: {nutrition.get('fats', 'N/A')}g
Fiber: {nutrition.get('fiber', 'N/A')}g"""

            properties = result.get("property", {})
            meal_types = []
            if properties.get("isBreakfast"):
                meal_types.append("Breakfast")
            if properties.get("isLunch"):
                meal_types.append("Lunch")
            if properties.get("isDinner"):
                meal_types.append("Dinner")
            if properties.get("isSnack"):
                meal_types.append("Snack")

            categories = result.get("categories", [])
            category_names = [
                CATEGORY_LABELS.get(c, f"Category {c}") for c in categories
            ]

            content = f"""Food Name: {result.get('name', 'Unknown')}

Description:
{result.get('text_content', 'No description available')}

Nutritional Information:
{nutrition_text}

Meal Types: {', '.join(meal_types) if meal_types else 'N/A'}
Categories: {', '.join(category_names)}
Cooking Time: {properties.get('totalTime', 'N/A')} minutes
Complexity: {properties.get('complexity', 'N/A')}"""

            doc = Document(
                page_content=content.strip(),
                metadata={
                    "name": result.get("name"),
                    "score": result.get("score", 0),
                    "categories": category_names,
                    "nutrition": nutrition,
                    "properties": properties,
                    "source": "food_database",
                },
            )
            documents.append(doc)

        # Step 4: Re-rank if we have a semantic query and results
        if query and documents:
            documents = self.rerank_results(query, documents)

        logger.info(LogMessages.FOOD_RAG_RESULT.format(query=query, names=[d.metadata['name'] for d in documents]))
        return documents

    def rerank_results(self, query: str, documents: List[Document]) -> List[Document]:
        """Re-rank documents using Cross-Encoder"""
        if not documents:
            return []

        # Prepare pairs: (Query, Document Text)
        pairs = [[query, doc.page_content] for doc in documents]

        # Predict scores
        scores = self.cross_encoder.predict(pairs)

        # Attach scores and sort
        for i, doc in enumerate(documents):
            doc.metadata["rerank_score"] = float(scores[i])

        # Sort by rerank_score (descending)
        documents.sort(key=lambda x: x.metadata["rerank_score"], reverse=True)

        return documents

    def close(self):
        """Close MongoDB connection"""
        self.client.close()
