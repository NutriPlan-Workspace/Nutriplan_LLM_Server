from typing import List, Dict
from llm_agent.rag.food_rag import FoodRAGPipeline
from llm_agent.rag.manual_rag import UserManualRAGPipeline
from llm_agent.agent import MealPlannerAgent
from llm_agent.utils.messages import LogMessages
from llm_agent.utils.logger import logger

class CompleteMealPlannerPipeline:
    """Complete pipeline integrating all components"""

    def __init__(
        self,
        mongodb_uri: str,
        database_name: str,
        vllm_api_key: str,
        vllm_base_url: str,
        vllm_model_name: str,
        vllm_temperature: float,
        backend_url: str,
        embedding_model_name: str,
        food_collection: str = "foods",
        manual_collection: str = "llm_documents",
    ):
        logger.info(LogMessages.PIPELINE_INIT)

        self.food_rag_pipeline = FoodRAGPipeline(
            mongodb_uri=mongodb_uri,
            database_name=database_name,
            collection_name=food_collection,
            model_name=embedding_model_name,
        )

        self.manual_rag_pipeline = UserManualRAGPipeline(
            mongodb_uri=mongodb_uri,
            database_name=database_name,
            collection_name=manual_collection,
            model_name=embedding_model_name,
        )

        self.agent = MealPlannerAgent(
            food_rag=self.food_rag_pipeline,
            manual_rag=self.manual_rag_pipeline,
            vllm_base_url=vllm_base_url,
            vllm_model_name=vllm_model_name,
            vllm_api_key=vllm_api_key,
            vllm_temperature=vllm_temperature,
            backend_url=backend_url,
        )

        logger.info(LogMessages.PIPELINE_READY)

    def chat(self, message: str, chat_history: List[Dict] = None, auth_token: str = None):
        """Main chat interface (Generator)"""
        return self.agent.chat(message, chat_history, auth_token)

    def search_food(self, query: str, k: int = 3, filters: Dict = None):
        """Direct food search"""
        return self.food_rag_pipeline.search(query, k, filters)

    def search_manual(self, query: str, top_k: int = 3, filters: Dict = None):
        """
        Direct manual search

        Returns:
            List of dicts with 'text', 'metadata', and 'score'
        """
        return self.manual_rag_pipeline.semantic_search(query, top_k, filters)

    def close(self):
        """Close all connections"""
        self.food_rag_pipeline.close()
        self.manual_rag_pipeline.close()
        logger.info(LogMessages.PIPELINE_CLOSED)
