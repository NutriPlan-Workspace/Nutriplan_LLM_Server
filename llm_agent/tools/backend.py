import os
import requests
from llm_agent.utils.logger import logger
from llm_agent.utils.messages import LogMessages

class BackendDataTool:
    """Tool for fetching data from NestJS Backend"""

    def __init__(self, base_url: str):
        # Allow override via environment variable for WSL compatibility
        # In WSL, localhost doesn't reach Windows host; use host IP or set BACKEND_URL env var
        self.base_url = base_url
        logger.info(LogMessages.BACKEND_TOOL_INIT.format(url=self.base_url))

    def get_pantry_items(self, token: str, status: str = "in_pantry") -> str:
        """Fetch pantry items"""
        try:
            headers = {
                "Authorization": f"Bearer {token}",
                "Cookie": f"accessToken={token}",
                "User-Agent": "NutriPlan-LLM-Agent/1.0"
            }
            # Log headers for debugging (FULL)
            logger.debug(LogMessages.BACKEND_TOOL_HEADERS.format(headers=headers))
            response = requests.get(
                f"{self.base_url}/pantry",
                headers=headers,
                params={"status": status, "limit": 20},
            )
            if response.status_code == 200:
                data = response.json().get("data", [])
                if not data:
                    return f"Pantry ({status}) is empty."

                items = [f"- {item['name']} ({item['quantity']} {item.get('unit', '')})" for item in data]
                result = f"Items in {status}:\n" + "\n".join(items)
                logger.info(LogMessages.BACKEND_TOOL_PANTRY_RESULT.format(result=result))
                return result
            logger.error(LogMessages.BACKEND_TOOL_PANTRY_ERROR.format(status=response.status_code))
            return f"Error fetching pantry: {response.status_code}"
        except Exception as e:
            return f"Backend connection error: {e}"

    def get_user_profile(self, token: str) -> str:
        """Fetch user profile"""
        try:
            headers = {
                "Authorization": f"Bearer {token}",
                "Cookie": f"accessToken={token}",
                "User-Agent": "NutriPlan-LLM-Agent/1.0"
            }
            # Log headers for debugging (FULL)
            logger.debug(LogMessages.BACKEND_TOOL_HEADERS.format(headers=headers))
            response = requests.get(f"{self.base_url}/user/me", headers=headers)
            if response.status_code == 200:
                data = response.json().get("data", {})
                logger.info(LogMessages.BACKEND_TOOL_PROFILE_RESULT.format(data=data))
                return f"User Profile: {data.get('fullName')} ({data.get('email')})"
            logger.error(LogMessages.BACKEND_TOOL_PROFILE_ERROR.format(status=response.status_code))
            return f"Error fetching profile: {response.status_code}"
        except Exception as e:
            return f"Backend connection error: {e}"

    def get_daily_plan(self, token: str, date: str) -> str:
        """Fetch daily meal plan"""
        logger.info(LogMessages.BACKEND_TOOL_FETCHING_PLAN.format(url=self.base_url, date=date))
        if token: logger.debug(f"[BackendTool] Token: {token[:10]}...")
        else: logger.warning("[BackendTool] NO TOKEN")

        try:
            headers = {
                "Authorization": f"Bearer {token}",
                "Cookie": f"accessToken={token}",
                "User-Agent": "NutriPlan-LLM-Agent/1.0"
            }
            # Log headers for debugging (FULL)
            logger.debug(LogMessages.BACKEND_TOOL_HEADERS.format(headers=headers))

            # Correct endpoint based on FE code: /planner?date=YYYY-MM-DD
            # We use headers for Cookie to ensure exact formatting
            response = requests.get(
                f"{self.base_url}/planner",
                headers=headers,
                params={"date": date}
            )
            logger.debug(f"[BackendTool] Status Code: {response.status_code}")
            if response.status_code == 200:
                data = response.json().get("data", {})
                logger.debug(f"[BackendTool] Data Type: {type(data)}")
                logger.debug(f"[BackendTool] Data Content: {data}")

                if isinstance(data, list):
                    if len(data) > 0:
                        data = data[0]
                    else:
                        logger.info("[BackendTool] Data is an empty list.")
                        return f"No meal plan found for {date}."

                if not data:
                     return f"No meal plan found for {date}."

                # Summarize the meal plan
                summary = []
                meal_items = data.get("mealItems", {})
                for meal_type, items in meal_items.items():
                    if not items: continue
                    item_names = [f"{i.get('foodId', {}).get('name', 'Unknown')}" for i in items]
                    summary.append(f"{meal_type.capitalize()}: {', '.join(item_names)}")

                if not summary:
                    logger.info(LogMessages.BACKEND_TOOL_PLAN_EMPTY.format(date=date))
                    return f"Meal plan for {date} is empty."

                result = f"Meal Plan for {date}:\n" + "\n".join(summary)
                logger.info(LogMessages.BACKEND_TOOL_PLAN_RESULT.format(date=date, result=result))
                return result
            logger.error(f"[BackendTool] Error Body: {response.text}")
            return f"Error fetching meal plan: {response.status_code} - {response.text}"
        except Exception as e:
            import traceback
            logger.error(f"[BackendTool] EXCEPTION: {e}")
            logger.error(traceback.format_exc())
            return f"Backend connection error: {e}"
