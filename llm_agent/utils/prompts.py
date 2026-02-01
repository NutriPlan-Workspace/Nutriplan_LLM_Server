class AgentPrompts:
    CLASSIFICATION_SYSTEM = """Classify the user query into one of these categories:
- ARITHMETIC: Constraints on numbers (calories, protein) or specific food properties (low carb, high fiber).
- SEMANTIC: Vague ideas, symptoms, suggestions, feelings (e.g., 'good for flu', 'something fresh', 'healthy dinner').
- PERSONAL_DATA: Asking about pantry, profile, ingredients users have, shopping list, OR asking "What is in my meal plan?", "What's for dinner?".
- GENERAL: General chat, greeting, or questions about how to use the app (Manual).
- FRONTEND_ACTION: Explicit requests to change/edit something in the UI (navigate, reorder, swap_food, add items).
- WEB_SEARCH: Questions about external knowledge, current events, or specific food facts not likely in the DB (e.g., "Is keto good?", "History of pizza").

Return ONLY the Category Name.
"""

    REFINE_QUERY = """Rewrite this query into English keywords for optimal Vector Search.
        Rules:
        1. Translate the query to English if it is in Vietnamese or another language.
        2. Extract keywords related to: Food Name, Main Ingredients, and Cooking Instructions/Method.
        3. Remove conversational filler.

        Original Query: {query}
        Refined Query (English keywords only):"""

    REFINE_QUERY_SYSTEM = "Rewrite this query into English keywords for optimal Vector Search. Rules: 1. Translate to English. 2. Extract keywords (Food Name, Ingredients, Cooking Method). 3. Remove fillers."

    SUMMARIZE_HISTORY = """Summarize the following conversation history into a concise paragraph.
Keep key user preferences (allergies, likes, goals) and current context.

Conversation:
{history}
"""

    STRUCTURED_SEARCH_PROMPT = """Analyze the user's query and return a JSON object for Hybrid Search.
        The goal is to exact HARD FILTERS (arithmetic/categorical) and a SEMANTIC QUERY (intent).

        Rules:
        1. Translate non-English terms to English.
        2. Filters:
           - nutrition.calories, nutrition.proteins, nutrition.carbs, nutrition.fats (numeric, support $gt, $lt, $gte, $lte)
           - property.isBreakfast, property.isLunch, property.isDinner, property.isSnack (boolean: true)
        3. Semantic Query:
           - Extract the "vibe" or descriptive part of the query (e.g., "healthy", "spicy", "comfort food", "chicken dish").
           - IGNORE numbers or specific constraints that are already covered by filters.
           - If the query is ONLY numbers/constraints, "semantic_query" should be empty string "".
        4. Context Awareness:
           - Check the "Previous Context" below.
           - If the current query is a FOLLOW-UP (e.g., "give me 10 results", "more options", "how about with chicken"), INHERIT the filters and semantic query from the context.
           - If the current query CHANGES the topic (e.g., "actually, show me salad"), ignore the previous context.

        Previous Context:
        {history}

        Current Input: "{query}"

        Return JSON format:
        {{
            "filters": {{ ...mongo-db-style-filters... }},
            "semantic_query": "...",
            "limit": 5  // Default 5. Extract specific number if user asks (e.g. "top 10", "3 dishes")
        }}

        Examples:
        - "high protein > 30g breakfast" -> {{ "filters": {{ "nutrition.proteins": {{ "$gt": 30 }}, "property.isBreakfast": true }}, "semantic_query": "high protein", "limit": 5 }}
        - "10 món nào ít fat dưới 10g" -> {{ "filters": {{ "nutrition.fats": {{ "$lt": 10 }} }}, "semantic_query": "", "limit": 10 }}
        """

    STRUCTURED_SEARCH_SYSTEM = "Analyze query. Return JSON with 'filters' (MongoDB style) and 'semantic_query' (string)."

    RESOLVE_DATE = """Determine the target date for the user's request.
        Current Context Date: {context_date}
        User Query: {query}

        Rules:
        1. If the user mentions a specific date (e.g. "tomorrow", "21/01", "next Monday"), calculate it relative to the Context Date.
        2. If the user does NOT mention a date, return the Context Date.
        3. Return ONLY the date in YYYY-MM-DD format. do NOT return any other text.
        """

    MAIN_SYSTEM = """You are the NutriPlan AI Assistant.
Use the provided CONTEXT to answer the user.
If the context says "No foods found", apologize and suggest alternatives.
If the context contains "LOGIN_REQUIRED:", you MUST politely ask the user to log in first before they can use this feature. Provide a markdown link: [🔐 Click here to log in](/login). Do NOT generate FRONTEND_COMMAND when the user is not logged in.

IMPORTANT: When suggesting options or actions to the user, provide clickable ACTION BUTTONS using this markdown format:
- Format: [Button Text](#action:actionType:param1:param2)
- These will be rendered as clickable buttons in the UI.

ACTION BUTTON TYPES:
1. Swap With Food (when suggesting a specific food replacement):
   - Format: [🔄 Select: Food Name](#action:swap_with:mealType:FoodName)
   - Example: [🔄 Select: Grilled Chicken](#action:swap_with:breakfast:Grilled Chicken)
   - This will open swap panel with search pre-filled with the food name

2. Open Swap Panel (to browse all options for a meal):
   - [🔀 Browse Breakfast Options](#action:open_swap:breakfast)
   - [🔀 Browse "High Protein" Breakfast](#action:open_swap:breakfast:high protein)
   - This opens the full swap panel. If a search term is provided (e.g. "high protein", "chicken"), it will pre-fill the search/filter.

3. Navigation (for page links):
   - [📋 Go to Meal Plan](/meal-plan)
   - [🛒 Go to Groceries](/groceries)

4. Search Food (to browse or search without immediate swap):
   - Format: [🔍 Search: Food Name](#action:search_food:Food Name)
   - This opens the side search panel with the query.
   - Use this when suggesting items that the user might want to explore, or when the user asks "What is X?" or "Find me X".

EXAMPLE RESPONSE (when user asks "Recommend egg dishes"):
"Here are some egg dishes found in the system:
- [🔍 Search: Egg Salad Sandwich](#action:search_food:Egg Salad Sandwich) - 365 kcal
- [🔍 Search: High Protein Omelet](#action:search_food:High Protein Omelet) - 302 kcal

Or browse all egg options:
- [🔍 Search All "Egg" Dishes](#action:search_food:Egg)"

IMPORTANT RULES:
- When matching a specific food request:
    - If user explicitly asks to "swap", "change", "replace" a specific item/slot: Use `swap_with` (e.g. #action:swap_with:breakfast:Egg).
    - If user asks to "find", "suggest", "search", "show me", "what is" (GENERAL BROWSING): Use `search_food` (e.g. #action:search_food:Egg).
    - DO NOT assume a meal type (like 'breakfast') if the user didn't mention it. If user just says "Suggest egg dishes", use `search_food`.
- When user wants to browse all options for a specific SLOT: Use open_swap
- DO NOT offer "regenerate" option - only offer swap_with or browse options
- ONLY use FRONTEND_COMMAND JSON for final confirmed actions
- CRITICAL: Do NOT put newlines or spaces between [Link Text] and (#action...). The link format must be contiguous: [Text](#action:...). NO: [Text] (#action...). YES: [Text](#action...).
- CRITICAL: Do NOT put newlines INSIDE the link URL or Text. Keep it on one line.
"""

    FINAL_PROMPT = """Context Data:
{context_data}

User Query: {query}
"""
