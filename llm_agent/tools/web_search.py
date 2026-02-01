from ddgs import DDGS
from llm_agent.utils.logger import logger
from llm_agent.utils.messages import LogMessages

class WebSearchTool:
    """Tool for searching the web using DuckDuckGo"""

    def __init__(self):
        logger.info(LogMessages.WEB_SEARCH_INIT)
        try:
            self.ddgs = DDGS()
            logger.info(LogMessages.WEB_SEARCH_SUCCESS)
        except Exception as e:
            logger.error(LogMessages.WEB_SEARCH_FAIL.format(error=e))
            self.ddgs = None

    def search(self, query: str, max_results: int = 3) -> str:
        """Perform a web search"""
        if not self.ddgs:
            return "Web search is unavailable."

        logger.info(LogMessages.WEB_SEARCH_QUERY.format(query=query))
        try:
            results = list(self.ddgs.text(query, max_results=max_results))
            if not results:
                return "No web results found."

            formatted = []
            for r in results:
                formatted.append(f"Title: {r.get('title')}\nLink: {r.get('href')}\nSnippet: {r.get('body')}\n")

            summary = "\n---\n".join(formatted)
            logger.info(LogMessages.WEB_SEARCH_RESULT.format(query=query, result=summary))
            return summary
        except Exception as e:
            logger.error(LogMessages.WEB_SEARCH_ERROR.format(error=e))
            return f"Error performing web search: {e}"
