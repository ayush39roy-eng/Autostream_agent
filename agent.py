"""
agent.py — AutoStream Conversational AI Agent

A stateful conversational agent built with LangGraph + Groq (Llama 3 8B) that:
1. Detects user intent (casual_greeting, product_inquiry, high_intent)
2. Answers product questions via RAG from a local knowledge base
3. Captures leads by collecting name, email, and platform step-by-step
4. Maintains conversation state across multiple turns

Architecture:
    The agent uses a LangGraph StateGraph to model conversation flow as a
    directed graph. Each node represents a processing step (classify intent,
    retrieve knowledge, collect lead info, etc.) and edges define transitions
    based on the detected intent and current state.
"""

import json
import os
import time
from typing import Annotated, TypedDict, Literal, Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

from tools import mock_lead_capture

# ---------------------------------------------------------------------------
# Environment setup — load API key from .env or environment variable
# ---------------------------------------------------------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. Set it in your environment or .env file."
    )

# ---------------------------------------------------------------------------
# Initialize Groq with Llama 3 8B via LangChain's Groq integration
# Groq offers free API access with generous rate limits (30 RPM, 14400 RPD)
# ---------------------------------------------------------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY,
    temperature=0.3,       # Low temperature for consistent, factual responses
    max_tokens=512,        # Keep output concise
)

# ---------------------------------------------------------------------------
# Rate-limit throttle delay (seconds) inserted before every LLM call
# Groq free tier allows 30 RPM — 3s delay keeps us well under the limit
# ---------------------------------------------------------------------------
THROTTLE_DELAY = 3

# ---------------------------------------------------------------------------
# Load knowledge base from JSON file for RAG retrieval
# This acts as our local vector-store substitute — simple keyword matching
# ---------------------------------------------------------------------------
KB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_base.json")

with open(KB_PATH, "r") as f:
    KNOWLEDGE_BASE = json.load(f)


def retrieve_knowledge(query: str) -> str:
    """
    Simple RAG retrieval: search the knowledge base for relevant information
    based on keyword matching against the user's query.

    In production, this would use a proper vector store (e.g., FAISS, Pinecone)
    with embeddings for semantic search. For this implementation, we use
    keyword matching which is sufficient for our small, structured knowledge base.

    Args:
        query: The user's question/message to search against.

    Returns:
        A formatted string containing all relevant knowledge base entries.
    """
    query_lower = query.lower()
    results = []

    # Search through pricing plans
    for plan in KNOWLEDGE_BASE.get("plans", []):
        # Check if the query mentions this plan or general pricing terms
        plan_name_lower = plan["name"].lower()
        if (
            plan_name_lower in query_lower
            or "price" in query_lower
            or "pricing" in query_lower
            or "plan" in query_lower
            or "cost" in query_lower
            or "how much" in query_lower
            or "subscription" in query_lower
        ):
            features_str = ", ".join(plan["features"])
            results.append(
                f"**{plan['name']}**: {plan['price']} — {features_str}"
            )

    # Search through company policies
    for policy in KNOWLEDGE_BASE.get("policies", []):
        policy_topic_lower = policy["topic"].lower()
        if (
            policy_topic_lower in query_lower
            or "refund" in query_lower
            or "support" in query_lower
            or "policy" in query_lower
            or "cancel" in query_lower
        ):
            results.append(f"**{policy['topic']}**: {policy['details']}")

    # If no specific matches, return a general company overview
    if not results:
        company = KNOWLEDGE_BASE.get("company", {})
        results.append(
            f"**{company.get('name', 'AutoStream')}**: "
            f"{company.get('description', 'Automated video editing tools for content creators.')}"
        )

    return "\n\n".join(results)


# ---------------------------------------------------------------------------
# State definition — TypedDict that LangGraph uses to track conversation
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    """
    The state that persists across all nodes in the LangGraph.

    Attributes:
        messages: Full conversation history (human + AI messages) for context.
        current_intent: The classified intent of the latest user message.
        lead_name: Collected lead name (None until provided).
        lead_email: Collected lead email (None until provided).
        lead_platform: Collected lead platform (None until provided).
        lead_capture_step: Tracks which piece of info to ask for next.
            Values: "idle" | "ask_name" | "ask_email" | "ask_platform" | "complete"
        response: The agent's response to send back to the user.
    """
    messages: list
    current_intent: str
    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]
    lead_capture_step: str
    response: str


# ---------------------------------------------------------------------------
# Node 1: Intent Classification
# Uses Groq/Llama to classify the user's message into one of three categories
# ---------------------------------------------------------------------------
def classify_intent(state: AgentState) -> AgentState:
    """
    Classify the user's latest message into one of three intents:
    - casual_greeting: Small talk, hellos, how are you, etc.
    - product_inquiry: Questions about pricing, features, plans, policies
    - high_intent: User wants to sign up, try, buy, or start using the product

    If the agent is currently in the middle of collecting lead information
    (lead_capture_step is not "idle" or "complete"), we skip classification
    and treat the message as continued lead data collection.
    """
    # If we're in the middle of collecting lead info, skip intent classification
    # The user's message is likely a response to our lead capture question
    if state["lead_capture_step"] not in ("idle", "complete"):
        state["current_intent"] = "collecting_lead_info"
        return state

    # Get the latest user message for classification
    latest_message = state["messages"][-1].content if state["messages"] else ""

    # Trimmed system prompt — minimal tokens for classification only
    classification_prompt = [
        SystemMessage(content=(
            "Classify this message into ONE category. "
            "Reply with ONLY the category name:\n"
            "- casual_greeting\n"
            "- product_inquiry\n"
            "- high_intent"
        )),
        HumanMessage(content=latest_message),
    ]

    # Throttle before LLM call to respect free-tier rate limits
    time.sleep(THROTTLE_DELAY)
    result = llm.invoke(classification_prompt)
    intent = result.content.strip().lower()

    # Validate the classification; default to product_inquiry if unclear
    valid_intents = {"casual_greeting", "product_inquiry", "high_intent"}
    if intent not in valid_intents:
        intent = "product_inquiry"

    state["current_intent"] = intent
    return state


# ---------------------------------------------------------------------------
# Node 2: Handle Casual Greeting
# Responds to greetings with a friendly, on-brand message
# ---------------------------------------------------------------------------
def handle_greeting(state: AgentState) -> AgentState:
    """
    Generate a warm, friendly greeting that introduces AutoStream
    and invites the user to ask questions about the product.
    """
    # Trimmed prompt + only last 4 messages to reduce token usage
    system_msg = SystemMessage(content=(
        "You are AutoStream's friendly assistant (video editing SaaS). "
        "Greet warmly in 2-3 sentences. Mention you can help with pricing or getting started."
    ))

    # Send only the last 4 messages to keep token count low
    recent_messages = state["messages"][-4:]
    time.sleep(THROTTLE_DELAY)
    result = llm.invoke([system_msg] + recent_messages)
    state["response"] = result.content
    # Append AI response to message history for context retention
    state["messages"].append(AIMessage(content=result.content))
    return state


# ---------------------------------------------------------------------------
# Node 3: Handle Product Inquiry (RAG)
# Retrieves relevant info from knowledge base and generates response
# ---------------------------------------------------------------------------
def handle_product_inquiry(state: AgentState) -> AgentState:
    """
    Answer product questions using RAG (Retrieval-Augmented Generation).

    1. Extract the user's question from the latest message
    2. Retrieve relevant entries from the knowledge base
    3. Feed the retrieved context + question to Gemini for a natural answer
    """
    latest_message = state["messages"][-1].content

    # Step 1: Retrieve relevant knowledge base entries
    kb_context = retrieve_knowledge(latest_message)

    # Step 2: Trimmed prompt + only last 4 messages to reduce token usage
    system_msg = SystemMessage(content=(
        "Answer using ONLY this info. Be concise.\n"
        f"{kb_context}"
    ))

    recent_messages = state["messages"][-4:]
    time.sleep(THROTTLE_DELAY)
    result = llm.invoke([system_msg] + recent_messages)
    state["response"] = result.content
    state["messages"].append(AIMessage(content=result.content))
    return state


# ---------------------------------------------------------------------------
# Node 4: Handle High Intent — Start Lead Capture Flow
# Detects signup intent and initiates the step-by-step info collection
# ---------------------------------------------------------------------------
def handle_high_intent(state: AgentState) -> AgentState:
    """
    When high_intent is detected, begin the lead capture workflow.
    Set the lead_capture_step to 'ask_name' and ask the user for their name.
    This is the entry point — subsequent turns go through collect_lead_info.
    """
    state["lead_capture_step"] = "ask_name"
    response = (
        "That's awesome! I'd love to help you get started with AutoStream. 🚀\n\n"
        "Let me get a few details so we can set things up for you.\n\n"
        "First, could you please tell me your **full name**?"
    )
    state["response"] = response
    state["messages"].append(AIMessage(content=response))
    return state


# ---------------------------------------------------------------------------
# Node 5: Collect Lead Info — Progressively gather name, email, platform
# ---------------------------------------------------------------------------
def collect_lead_info(state: AgentState) -> AgentState:
    """
    Progressively collect lead information one field at a time.
    
    Flow: ask_name → ask_email → ask_platform → complete
    
    Each step stores the user's latest message as the answer to the
    current question, then advances to the next step. Only when ALL
    three fields are collected does mock_lead_capture() get called.
    """
    latest_message = state["messages"][-1].content.strip()
    step = state["lead_capture_step"]

    if step == "ask_name":
        # Store the name and advance to email collection
        state["lead_name"] = latest_message
        state["lead_capture_step"] = "ask_email"
        response = f"Great to meet you, **{latest_message}**! 👋\n\nWhat's your **email address**?"
        state["response"] = response
        state["messages"].append(AIMessage(content=response))

    elif step == "ask_email":
        # Store the email and advance to platform collection
        state["lead_email"] = latest_message
        state["lead_capture_step"] = "ask_platform"
        response = (
            "Perfect, got it! ✉️\n\n"
            "Last question — what **content creation platform** do you primarily use? "
            "(e.g., YouTube, Instagram, TikTok, Twitch)"
        )
        state["response"] = response
        state["messages"].append(AIMessage(content=response))

    elif step == "ask_platform":
        # Store the platform — ALL 3 fields now collected, trigger lead capture
        state["lead_platform"] = latest_message
        state["lead_capture_step"] = "complete"

        # ✅ Call mock_lead_capture ONLY after all 3 fields are collected
        capture_result = mock_lead_capture(
            name=state["lead_name"],
            email=state["lead_email"],
            platform=state["lead_platform"],
        )

        response = (
            f"🎉 You're all set, **{state['lead_name']}**!\n\n"
            f"Here's a summary of your registration:\n"
            f"- **Name:** {state['lead_name']}\n"
            f"- **Email:** {state['lead_email']}\n"
            f"- **Platform:** {state['lead_platform']}\n\n"
            f"Our team will reach out shortly with next steps to get you started "
            f"on AutoStream. Welcome aboard! 🚀\n\n"
            f"Is there anything else I can help you with?"
        )
        state["response"] = response
        state["messages"].append(AIMessage(content=response))

    return state


# ---------------------------------------------------------------------------
# Router function — determines which node to visit based on intent
# ---------------------------------------------------------------------------
def route_by_intent(state: AgentState) -> str:
    """
    Routing function for LangGraph conditional edges.
    
    Returns the name of the next node based on the classified intent.
    This is called after classify_intent to direct the graph flow.
    """
    intent = state["current_intent"]

    if intent == "collecting_lead_info":
        # User is in the middle of providing lead details
        return "collect_lead_info"
    elif intent == "casual_greeting":
        return "handle_greeting"
    elif intent == "high_intent":
        return "handle_high_intent"
    else:
        # Default: product_inquiry or any unrecognized intent
        return "handle_product_inquiry"


# ---------------------------------------------------------------------------
# Build the LangGraph StateGraph
# ---------------------------------------------------------------------------
def build_graph() -> StateGraph:
    """
    Construct the LangGraph StateGraph that defines the agent's conversation flow.

    Graph structure:
        START → classify_intent → (router) → handle_greeting     → END
                                            → handle_product_inquiry → END
                                            → handle_high_intent  → END
                                            → collect_lead_info   → END

    Each conversation turn runs through this graph once. State is preserved
    between invocations to maintain memory across turns.
    """
    # Create a new StateGraph with our AgentState schema
    graph = StateGraph(AgentState)

    # Add nodes — each node is a function that transforms the state
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("handle_greeting", handle_greeting)
    graph.add_node("handle_product_inquiry", handle_product_inquiry)
    graph.add_node("handle_high_intent", handle_high_intent)
    graph.add_node("collect_lead_info", collect_lead_info)

    # Set the entry point — every conversation turn starts with intent classification
    graph.set_entry_point("classify_intent")

    # Add conditional edges from classify_intent to the appropriate handler
    # The route_by_intent function determines which node to visit next
    graph.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "handle_greeting": "handle_greeting",
            "handle_product_inquiry": "handle_product_inquiry",
            "handle_high_intent": "handle_high_intent",
            "collect_lead_info": "collect_lead_info",
        },
    )

    # All handler nodes terminate the graph after processing (one node per turn)
    graph.add_edge("handle_greeting", END)
    graph.add_edge("handle_product_inquiry", END)
    graph.add_edge("handle_high_intent", END)
    graph.add_edge("collect_lead_info", END)

    return graph


# ---------------------------------------------------------------------------
# Compile and run the agent in an interactive loop
# ---------------------------------------------------------------------------
def main():
    """
    Main entry point — compiles the graph and runs an interactive chat loop.

    State is initialized once and passed through the graph on each turn,
    ensuring conversation history and lead capture progress persist across
    multiple turns (5-6+ turns as required).
    """
    print("=" * 60)
    print("  🎬 AutoStream AI Assistant")
    print("  Automated Video Editing Tools for Content Creators")
    print("=" * 60)
    print("  Type 'quit' or 'exit' to end the conversation.\n")

    # Build and compile the LangGraph
    graph = build_graph()
    app = graph.compile()

    # Initialize conversation state — this persists across all turns
    state: AgentState = {
        "messages": [],
        "current_intent": "",
        "lead_name": None,
        "lead_email": None,
        "lead_platform": None,
        "lead_capture_step": "idle",  # Start in idle — no lead capture in progress
        "response": "",
    }

    # Interactive conversation loop
    turn_count = 0
    while True:
        # Get user input
        user_input = input("\n🧑 You: ").strip()

        # Exit conditions
        if user_input.lower() in ("quit", "exit", "q"):
            print("\n👋 Thanks for chatting with AutoStream! Goodbye!\n")
            break

        if not user_input:
            continue

        turn_count += 1

        # Append the user's message to conversation history
        state["messages"].append(HumanMessage(content=user_input))

        # Run the graph — processes one turn through the state machine
        # The graph modifies state in-place through the node functions
        state = app.invoke(state)

        # Display the agent's response
        print(f"\n🤖 AutoStream: {state['response']}")

        # Debug info (optional — shows detected intent for each turn)
        print(f"   [Turn {turn_count} | Intent: {state['current_intent']} | "
              f"Lead Step: {state['lead_capture_step']}]")


if __name__ == "__main__":
    main()
