class LogMessages:
    # Agent
    AGENT_INIT = "🤖 Initializing Meal Planner Agent..."
    AGENT_READY = "   ✓ Agent initialized with Router-based flow"
    AGENT_NEW_REQUEST = "[MealPlannerAgent] New Request: {message}"
    AGENT_SUMMARIZING_HISTORY = "[LLM Server] -> Summarizing long history..."
    AGENT_HISTORY_SUMMARIZED = "[MealPlannerAgent] History summarized"
    AGENT_CLASSIFICATION_START = "[LLM Server] -> Classifying query..."
    AGENT_CLASSIFICATION_RESULT = "[LLM Server] Category Result: {category}"
    AGENT_PATH_ARITHMETIC = "[LLM Server] -> PATH: ARITHMETIC"
    AGENT_PATH_SEMANTIC = "[LLM Server] -> PATH: SEMANTIC"
    AGENT_PATH_PERSONAL_DATA = "[LLM Server] -> PATH: PERSONAL_DATA"
    AGENT_PATH_PERSONAL_DATA_NO_AUTH = "[LLM Server] -> PATH: PERSONAL_DATA (NO AUTH)"
    AGENT_PATH_GENERAL = "[LLM Server] -> PATH: GENERAL"
    AGENT_PATH_WEB_SEARCH = "[LLM Server] -> PATH: WEB SEARCH"
    AGENT_PATH_FRONTEND_ACTION = "[LLM Server] -> PATH: FRONTEND_ACTION"

    # Branches
    AGENT_BRANCH_MEAL_PLAN = "[LLM Server] -> Branch: MEAL_PLAN"
    AGENT_BRANCH_PANTRY = "[LLM Server] -> Branch: PANTRY"
    AGENT_BRANCH_USER_PROFILE = "[LLM Server] -> Branch: USER_PROFILE"

    # Actions
    AGENT_FETCHING_MEAL_PLAN = "[LLM Server] Fetching meal plan for date: {date}"
    AGENT_RESOLVED_DATE = "[LLM Server] Resolved Target Date: {target_date} (Context: {context_date})"

    # Results
    AGENT_MONGO_SPEC = "[LLM Server] Mongo Spec: {spec}"
    AGENT_SEARCH_RESULT = "[LLM Server] Search Result:\n{result}"
    AGENT_REFINED_QUERY = "[LLM Server] Refined Query: {query}"
    AGENT_RAG_RESULTS = "[LLM Server] RAG Results ({count} docs):\n{result}"
    AGENT_MEAL_PLAN_RESULT = "[LLM Server] Meal Plan Result:\n{result}"
    AGENT_PANTRY_RESULT = "[LLM Server] Pantry Result:\n{result}"
    AGENT_PROFILE_RESULT = "[LLM Server] Profile Result:\n{result}"
    AGENT_MANUAL_SEARCH_RESULT = "[LLM Server] Manual Search Result: {result}..."
    AGENT_WEB_SEARCH_RESULT = "[LLM Server] Web Search Result: {result}..."

    # Generation
    AGENT_GENERATING_RESPONSE = "\n[LLM Server] === GENERATING RESPONSE ==="
    AGENT_CONTEXT_LENGTH = "[LLM Server] Context Data Length: {length} chars"
    AGENT_FINAL_PROMPT = "[LLM Server] Final Prompt to LLM:\n{prompt}..."
    AGENT_STARTING_STREAM = "[LLM Server] Starting LLM streaming..."

    # Tools
    BACKEND_TOOL_INIT = "[BackendDataTool] Initialized with base_url: {url}"
    BACKEND_TOOL_HEADERS = "[BackendTool] Request Headers: {headers}"
    BACKEND_TOOL_PANTRY_RESULT = "[BackendDataTool] Pantry Result:\n{result}"
    BACKEND_TOOL_PANTRY_ERROR = "[BackendDataTool] Error fetching pantry: {status}"
    BACKEND_TOOL_PROFILE_RESULT = "[BackendDataTool] Retrieved user profile: {data}"
    BACKEND_TOOL_PROFILE_ERROR = "[BackendDataTool] Error fetching profile: {status}"
    BACKEND_TOOL_FETCHING_PLAN = "\n[BackendTool] Fetching: {url}/planner | Date: {date}"
    BACKEND_TOOL_PLAN_RESULT = "[BackendDataTool] Retrieved meal plan for {date}:\n{result}"
    BACKEND_TOOL_PLAN_EMPTY = "[BackendDataTool] Meal plan empty for date: {date}"

    WEB_SEARCH_INIT = "🌐 Initializing Web Search Tool..."
    WEB_SEARCH_SUCCESS = "   ✓ DuckDuckGo Search initialized"
    WEB_SEARCH_FAIL = "   ❌ Failed to initialize DDGS: {error}"
    WEB_SEARCH_QUERY = "[WebSearch] Searching for: {query}"
    WEB_SEARCH_RESULT = "[WebSearchTool] Query: '{query}' -> Result:\n{result}"
    WEB_SEARCH_ERROR = "[WebSearch] Error: {error}"

    STRUCTURED_SEARCH_RESULT = "[StructuredFoodSearchTool] Spec: {spec} -> Result:\n{result}"

    # RAG
    FOOD_RAG_INIT = "🍕 Initializing Food RAG Pipeline..."
    FOOD_RAG_CONNECTED = "   ✓ Connected to {db}.{collection}"
    FOOD_RAG_COUNT = "   ✓ Total foods: {count}"
    FOOD_RAG_RESULT = "[FoodRAGPipeline] Query: '{query}' -> Retrieved: {names}"

    MANUAL_RAG_INIT = "📚 Initializing User Manual RAG Pipeline..."
    MANUAL_RAG_CONNECTED = "   ✓ Connected to {db}.{collection}"
    MANUAL_RAG_COUNT = "   ✓ Total chunks: {count}"
    MANUAL_RAG_RESULT = "[UserManualRAGPipeline] Semantic Search: '{query}' -> Results:\n{previews}"
    MANUAL_RAG_FAIL = "⚠️  Atlas Vector Search failed: {error}"

    # Pipeline
    PIPELINE_INIT = "=" * 70 + "\n🚀 INITIALIZING COMPLETE MEAL PLANNER PIPELINE\n" + "=" * 70
    PIPELINE_READY = "=" * 70 + "\n✅ PIPELINE READY\n" + "=" * 70
    PIPELINE_CLOSED = "✓ All connections closed"
