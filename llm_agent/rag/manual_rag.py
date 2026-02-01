from typing import List, Dict
import os
import torch
import numpy as np
from huggingface_hub import snapshot_download
from pymongo import MongoClient
from langchain_huggingface import HuggingFaceEmbeddings
from llm_agent.utils.logger import logger
from llm_agent.utils.messages import LogMessages

class UserManualRAGPipeline:
    """RAG Pipeline for User Manual"""

    def __init__(
        self,
        mongodb_uri: str,
        database_name: str,
        collection_name: str,
        model_name: str,
    ):
        logger.info(LogMessages.MANUAL_RAG_INIT)

        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.client = MongoClient(mongodb_uri)
        self.db = self.client[database_name]
        self.collection = self.db[collection_name]

        logger.info(LogMessages.MANUAL_RAG_CONNECTED.format(db=database_name, collection=collection_name))
        logger.info(LogMessages.MANUAL_RAG_COUNT.format(count=self.collection.count_documents({})))

    def semantic_search(
        self, query: str, top_k: int = 3, filters: Dict = None
    ) -> List[Dict]:
        """
        Semantic search in user manual

        Returns:
            List of dicts with 'text', 'metadata', and 'score' keys
        """
        query_vector = self.embeddings.embed_query(query)

        try:
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",
                        "path": "embedding",
                        "queryVector": query_vector,
                        "numCandidates": top_k * 10,
                        "limit": top_k,
                    }
                }
            ]

            if filters:
                pipeline.append({"$match": filters})

            pipeline.append(
                {
                    "$project": {
                        "text": 1,
                        "metadata": 1,
                        "score": {"$meta": "vectorSearchScore"},
                    }
                }
            )

            results = list(self.collection.aggregate(pipeline))

            previews = [r['text'][:100] + '...' for r in results]
            logger.info(LogMessages.MANUAL_RAG_RESULT.format(query=query, previews=previews))
            # Format results
            return [
                {"text": r["text"], "metadata": r["metadata"], "score": r["score"]}
                for r in results
            ]

        except Exception as e:
            logger.error(LogMessages.MANUAL_RAG_FAIL.format(error=e))
            logger.warning("   Falling back to manual search...")

            query_dict = filters if filters else {}
            all_docs = list(self.collection.find(query_dict))

            similarities = []
            for doc in all_docs:
                similarity = np.dot(query_vector, doc["embedding"])
                similarities.append(
                    {
                        "text": doc["text"],
                        "metadata": doc["metadata"],
                        "score": float(similarity),
                    }
                )

            similarities.sort(key=lambda x: x["score"], reverse=True)
            return similarities[:top_k]

    def close(self):
        """Close MongoDB connection"""
        self.client.close()
