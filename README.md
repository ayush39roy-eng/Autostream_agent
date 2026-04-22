#  AutoStream — Conversational AI Agent

A stateful conversational AI agent for **AutoStream**, a fictional SaaS company providing automated video editing tools for content creators. Built with **Python + LangGraph + Groq (Llama 3)**.

---

##  Features

| Capability | Description |
|---|---|
| **Intent Detection** | Classifies every message as `casual_greeting`, `product_inquiry`, or `high_intent` |
| **RAG Knowledge Base** | Answers pricing, feature, and policy questions from `knowledge_base.json` |
| **Lead Capture** | Collects Name → Email → Platform step-by-step, then calls `mock_lead_capture()` |
| **State Management** | Retains full conversation history and lead progress across 5-6+ turns |

---

##  How to Run Locally

### Prerequisites
- Python 3.9+
- A [Groq API Key](https://console.groq.com/keys) (free)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/autostream-agent.git
cd autostream-agent

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        
# venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your API key (choose one method)

# Option A: Create a .env file
echo "GROQ_API_KEY=your-api-key-here" > .env

# Option B: Export directly
export GROQ_API_KEY="your-api-key-here"

# 5. Run the agent
python agent.py
```

### Example Conversation

```
🧑 You: Hi, tell me about your pricing
🤖 AutoStream: We have two great plans! The Basic Plan is $29/month with 10 videos 
   and 720p resolution. The Pro Plan is $79/month with unlimited videos, 4K, and 
   AI-powered captions...

🧑 You: I want to try Pro for my YouTube channel
🤖 AutoStream: That's awesome! Let me get a few details...
   First, could you please tell me your full name?

🧑 You: Ayush Roy
🤖 AutoStream: Great to meet you, Ayush Roy! What's your email address?

🧑 You: ayush@example.com
🤖 AutoStream: Perfect! What content creation platform do you primarily use?

🧑 You: YouTube
🤖 AutoStream:  You're all set, Ayush Roy! Our team will reach out shortly...
```

---

##  Architecture Explanation 

### Why LangGraph?

LangGraph was chosen over a simple LangChain sequential chain because the AutoStream agent requires **stateful, multi-turn conversations with branching logic**. A content creator asking about pricing follows a fundamentally different path than one ready to sign up — LangGraph's **directed graph model** naturally represents these divergent flows as nodes and edges, making the logic explicit and maintainable.

### How State is Managed

The agent's state is defined as a `TypedDict` (`AgentState`) containing the full message history, current intent classification, lead capture fields (name, email, platform), and a `lead_capture_step` tracker. This state object is **initialized once** and **passed through the graph on every conversation turn**, ensuring memory persists across 5-6+ turns without external storage.

Each turn flows through the graph: **START → classify_intent → (router) → handler → END**. The router inspects the classified intent and the current `lead_capture_step` to direct flow to the correct node. When the user is mid-lead-capture, intent classification is bypassed entirely — the agent knows it's waiting for a specific data field, preventing premature tool invocation.

This architecture cleanly separates concerns (classification, retrieval, collection, tool execution) into discrete, testable nodes while LangGraph handles orchestration and state propagation.

---

##  WhatsApp Integration via Webhooks

To deploy this agent on WhatsApp, you would use the **WhatsApp Business API** (via Meta Cloud API or a provider like Twilio) with a webhook-based architecture:

### Architecture Overview

```
User (WhatsApp) → Meta Cloud API → Your Webhook Server → AutoStream Agent → Response → Meta Cloud API → User
```

### Step-by-Step Integration

1. **Set up a WhatsApp Business Account** on [Meta for Developers](https://developers.facebook.com/) and register a phone number.

2. **Create a Webhook Server** (e.g., using Flask or FastAPI) that listens for incoming messages:

   ```python
   from fastapi import FastAPI, Request
   import requests

   app = FastAPI()

   
   user_sessions = {}

   @app.post("/webhook")
   async def webhook(request: Request):
       data = await request.json()
       
       
       phone = data["entry"][0]["changes"][0]["value"]["messages"][0]["from"]
       message = data["entry"][0]["changes"][0]["value"]["messages"][0]["text"]["body"]
       
      
       if phone not in user_sessions:
           user_sessions[phone] = create_initial_state()
       
       
       state = user_sessions[phone]
       state["messages"].append(HumanMessage(content=message))
       state = agent_app.invoke(state)
       user_sessions[phone] = state
       send_whatsapp_message(phone, state["response"])
       
       return {"status": "ok"}
   ```

3. **Configure Meta Webhook**: Point `https://yourdomain.com/webhook` as the callback URL in the Meta Developer Dashboard with a verify token.

4. **Send replies via the WhatsApp API**:
   ```python
   def send_whatsapp_message(to: str, text: str):
       url = f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages"
       headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
       payload = {
           "messaging_product": "whatsapp",
           "to": to,
           "text": {"body": text},
       }
       requests.post(url, headers=headers, json=payload)
   ```

5. **Session Management**: Use **Redis** or a database to persist `AgentState` per user phone number across webhook invocations, since each HTTP request is stateless.

6. **Deploy**: Host on a cloud platform (AWS, GCP, Railway) with HTTPS and a public domain.

---

##  Project Structure

```
autostream-agent/
├── agent.py              # Main agent with LangGraph state graph
├── knowledge_base.json   # RAG data (plans, policies, company info)
├── tools.py              # mock_lead_capture() function
├── requirements.txt      # Python dependencies
└── README.md             # This file
```



