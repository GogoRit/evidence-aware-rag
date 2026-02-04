"""Streamlit frontend for RAG System."""

import os
from datetime import datetime

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Page config
st.set_page_config(
    page_title="RAG System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for distinctive styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Outfit:wght@300;400;600;700&display=swap');
    
    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --bg-tertiary: #1a1a25;
        --accent-cyan: #00d4ff;
        --accent-magenta: #ff006e;
        --accent-yellow: #ffd60a;
        --accent-green: #00ff88;
        --text-primary: #f0f0f5;
        --text-secondary: #8888aa;
    }
    
    .stApp {
        background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
    }
    
    .main-header {
        font-family: 'Outfit', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, var(--accent-cyan), var(--accent-magenta));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 0.5rem 0;
        margin-bottom: 1rem;
    }
    
    .stat-card {
        background: var(--bg-tertiary);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.15);
    }
    
    .stat-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.75rem;
        font-weight: 600;
        color: var(--accent-cyan);
    }
    
    .stat-label {
        font-family: 'Outfit', sans-serif;
        color: var(--text-secondary);
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    
    .chat-message {
        padding: 1rem 1.25rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        font-family: 'Outfit', sans-serif;
    }
    
    .user-message {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.15), rgba(0, 212, 255, 0.05));
        border-left: 3px solid var(--accent-cyan);
    }
    
    .assistant-message {
        background: linear-gradient(135deg, rgba(255, 0, 110, 0.1), rgba(255, 0, 110, 0.02));
        border-left: 3px solid var(--accent-magenta);
    }
    
    .refused-message {
        background: linear-gradient(135deg, rgba(255, 0, 110, 0.15), rgba(255, 0, 110, 0.05));
        border-left: 3px solid var(--accent-magenta);
    }
    
    .source-card {
        background: var(--bg-tertiary);
        border: 1px solid rgba(255, 214, 10, 0.2);
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.4rem 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
    }
    
    .info-panel {
        background: var(--bg-tertiary);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .info-row {
        display: flex;
        justify-content: space-between;
        padding: 0.4rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        font-family: 'Outfit', sans-serif;
        font-size: 0.9rem;
    }
    
    .info-label {
        color: var(--text-secondary);
    }
    
    .info-value {
        font-family: 'JetBrains Mono', monospace;
        color: var(--accent-cyan);
    }
    
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 0.75rem;
    }
    
    .mini-stat {
        background: rgba(0, 212, 255, 0.05);
        border: 1px solid rgba(0, 212, 255, 0.1);
        border-radius: 8px;
        padding: 0.75rem;
        text-align: center;
    }
    
    .mini-stat-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.1rem;
        color: var(--accent-cyan);
    }
    
    .mini-stat-label {
        font-size: 0.7rem;
        color: var(--text-secondary);
        text-transform: uppercase;
    }
    
    .stTextInput > div > div > input {
        background: var(--bg-tertiary) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: var(--text-primary) !important;
        font-family: 'Outfit', sans-serif !important;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, var(--accent-cyan), var(--accent-magenta)) !important;
        color: white !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.5rem !important;
        transition: transform 0.2s, box-shadow 0.2s !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(0, 212, 255, 0.3) !important;
    }
    
    .dashboard-card {
        background: var(--bg-tertiary);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.5rem 0;
    }
    
    .dashboard-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1rem;
        color: var(--text-secondary);
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
</style>
""", unsafe_allow_html=True)


# ============ API Functions ============

def check_api_health() -> dict | None:
    """Check if the API is healthy."""
    try:
        response = httpx.get(f"{API_BASE_URL}/health", timeout=5.0)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


def ingest_document(workspace_id: str, file) -> dict | None:
    """Upload and ingest a document."""
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        data = {"workspace_id": workspace_id}
        
        response = httpx.post(
            f"{API_BASE_URL}/ingest",
            files=files,
            data=data,
            timeout=60.0,
        )
        
        if response.status_code == 201:
            return response.json()
        else:
            st.error(f"Ingestion failed: {response.text}")
    except Exception as e:
        st.error(f"Error: {str(e)}")
    return None


def query_rag(workspace_id: str, query: str, top_k: int = 5) -> dict | None:
    """Query the RAG system."""
    try:
        response = httpx.post(
            f"{API_BASE_URL}/chat",
            json={
                "workspace_id": workspace_id,
                "query": query,
                "top_k": top_k,
                "include_sources": True,
            },
            timeout=120.0,
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Query failed: {response.text}")
    except Exception as e:
        st.error(f"Error: {str(e)}")
    return None


def get_metrics() -> dict | None:
    """Get system metrics."""
    try:
        response = httpx.get(f"{API_BASE_URL}/metrics", timeout=5.0)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


def get_workspaces() -> list[str]:
    """Get list of available workspaces."""
    try:
        response = httpx.get(f"{API_BASE_URL}/workspaces", timeout=5.0)
        if response.status_code == 200:
            return response.json().get("workspaces", [])
    except Exception:
        pass
    return []


# ============ UI Components ============

def render_evidence_panel(response: dict | None, show_full_citations: bool = False):
    """
    Render the Evidence panel with response details.
    
    Shows:
    - Confidence level and scores
    - Refused status and reason
    - Model used
    - Latency and cost
    - Sources list (ranked)
    """
    st.markdown("#### Evidence")
    
    # Empty state
    if response is None:
        st.markdown("""
        <div class="info-panel" style="text-align: center; padding: 2rem;">
            <p style="color: var(--text-secondary); margin: 0;">
                No response yet.<br>Send a query to view evidence.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    router = response.get("router_decision", {})
    confidence = response.get("confidence", {})
    
    # === Status Section ===
    is_refused = response.get("refused", False)
    is_retrieval_only = response.get("retrieval_only", False)
    
    if is_refused:
        st.markdown("""
        <div style="background: rgba(255, 0, 110, 0.1); border-left: 3px solid #ff006e; 
                    padding: 0.75rem; border-radius: 4px; margin-bottom: 1rem;">
            <strong style="color: #ff006e;">Refused</strong>
        </div>
        """, unsafe_allow_html=True)
        
        refusal_reason = response.get("refusal_reason", "Insufficient confidence")
        st.caption(f"Reason: {refusal_reason}")
    elif is_retrieval_only:
        st.markdown("""
        <div style="background: rgba(0, 212, 255, 0.1); border-left: 3px solid #00d4ff; 
                    padding: 0.75rem; border-radius: 4px; margin-bottom: 1rem;">
            <strong style="color: #00d4ff;">Retrieval Only</strong>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: rgba(0, 255, 136, 0.1); border-left: 3px solid #00ff88; 
                    padding: 0.75rem; border-radius: 4px; margin-bottom: 1rem;">
            <strong style="color: #00ff88;">Generated</strong>
        </div>
        """, unsafe_allow_html=True)
    
    # === Confidence Section ===
    st.markdown("**Confidence**")
    conf_level = confidence.get("level", "unknown")
    top_score = confidence.get("top_score", 0)
    mean_score = confidence.get("mean_score", 0)
    
    # Color based on level
    level_colors = {
        "high": "#00ff88",
        "low": "#ffd60a",
        "insufficient": "#ff006e",
    }
    level_color = level_colors.get(conf_level, "#8888aa")
    
    st.markdown(f"""
    <div class="info-panel">
        <div class="info-row">
            <span class="info-label">Level</span>
            <span style="color: {level_color}; font-weight: 600;">{conf_level.upper()}</span>
        </div>
        <div class="info-row">
            <span class="info-label">Top Score</span>
            <span class="info-value">{top_score:.1%}</span>
        </div>
        <div class="info-row" style="border-bottom: none;">
            <span class="info-label">Mean Score</span>
            <span class="info-value">{mean_score:.1%}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # === Model & Performance Section ===
    st.markdown("**Model & Performance**")
    model = router.get("chosen_model", response.get("model_used", "N/A"))
    mode = router.get("mode", "unknown")
    latency = response.get("latency_ms", 0)
    cost = response.get("cost_usd") or router.get("cost_estimate_usd", 0)
    
    st.markdown(f"""
    <div class="info-panel">
        <div class="info-row">
            <span class="info-label">Model</span>
            <span class="info-value">{model}</span>
        </div>
        <div class="info-row">
            <span class="info-label">Mode</span>
            <span class="info-value">{mode}</span>
        </div>
        <div class="info-row">
            <span class="info-label">Latency</span>
            <span class="info-value">{latency:.0f} ms</span>
        </div>
        <div class="info-row" style="border-bottom: none;">
            <span class="info-label">Cost</span>
            <span class="info-value">${cost:.4f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # === Sources Section ===
    sources = response.get("sources", [])
    if sources:
        st.markdown(f"**Sources** ({len(sources)} retrieved)")
        
        # Determine how many to show
        display_count = len(sources) if show_full_citations else min(3, len(sources))
        
        for i, source in enumerate(sources[:display_count]):
            rank = source.get("metadata", {}).get("rank", i + 1)
            score = source.get("score", 0)
            filename = source.get("metadata", {}).get("filename", None)
            doc_id = source.get("document_id", "N/A")
            content = source.get("content", "")
            
            # Use filename if available, otherwise doc_id
            source_label = filename if filename else f"{doc_id[:12]}..."
            
            # Get a short snippet (1-2 lines)
            snippet = content[:120].replace("\n", " ").strip()
            if len(content) > 120:
                snippet += "..."
            
            # Score color based on value
            score_color = "#00ff88" if score >= 0.4 else "#ffd60a" if score >= 0.25 else "#ff006e"
            
            st.markdown(f"""
            <div class="source-card">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.4rem;">
                    <span style="color: var(--text-secondary);">#{rank}</span>
                    <span style="color: {score_color}; font-weight: 600;">{score:.1%}</span>
                </div>
                <div style="font-size: 0.85rem; color: var(--accent-cyan); margin-bottom: 0.3rem;">
                    {source_label}
                </div>
                <div style="font-size: 0.75rem; color: var(--text-secondary); line-height: 1.4;">
                    {snippet}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Show indicator if there are more sources
        if not show_full_citations and len(sources) > 3:
            st.caption(f"+ {len(sources) - 3} more sources (enable 'Show full citations')")


def render_chat_message(message: dict):
    """Render a single chat message in the chat history."""
    if message["role"] == "user":
        st.markdown(
            f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        is_refused = message.get("refused", False)
        is_retrieval_only = message.get("retrieval_only", False)
        
        if is_refused:
            refusal_reason = message.get("refusal_reason", "Insufficient confidence")
            st.markdown(
                f"""<div class="chat-message refused-message">
                <strong>Unable to answer</strong><br>
                <span style="opacity: 0.8; font-size: 0.9rem;">{refusal_reason}</span>
                </div>""",
                unsafe_allow_html=True,
            )
        elif is_retrieval_only:
            # For retrieval-only, show a brief confirmation in chat
            # Full details go in the Evidence panel
            confidence = message.get("confidence", {})
            top_score = confidence.get("top_score", 0)
            source_count = len(message.get("sources", []))
            
            st.markdown(
                f"""<div class="chat-message assistant-message">
                <strong>Retrieved {source_count} relevant sources</strong><br>
                <span style="opacity: 0.8; font-size: 0.9rem;">
                Top confidence: {top_score:.1%} — See Evidence panel for details
                </span>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            content = message.get("content") or "No answer generated."
            st.markdown(
                f'<div class="chat-message assistant-message"><strong>Assistant:</strong> {content}</div>',
                unsafe_allow_html=True,
            )


def render_dashboard(metrics: dict):
    """Render the dashboard page."""
    st.markdown("## System Dashboard")
    
    # Top stats row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{metrics.get('active_workspaces', 0)}</div>
            <div class="stat-label">Workspaces</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{metrics.get('documents_ingested', 0)}</div>
            <div class="stat-label">Documents</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{metrics.get('chunks_indexed', 0)}</div>
            <div class="stat-label">Chunks Indexed</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{metrics.get('requests_total', 0)}</div>
            <div class="stat-label">Total Requests</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Two columns for latency and costs
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("""
        <div class="dashboard-card">
            <div class="dashboard-title">Latency Metrics</div>
        </div>
        """, unsafe_allow_html=True)
        
        latency = metrics.get("latency")
        if latency:
            l1, l2, l3, l4 = st.columns(4)
            l1.metric("P50", f"{latency.get('p50_ms', 0):.0f}ms")
            l2.metric("P90", f"{latency.get('p90_ms', 0):.0f}ms")
            l3.metric("P99", f"{latency.get('p99_ms', 0):.0f}ms")
            l4.metric("Avg", f"{latency.get('avg_ms', 0):.0f}ms")
        else:
            st.info("No latency data yet")
    
    with col_right:
        st.markdown("""
        <div class="dashboard-card">
            <div class="dashboard-title">Cost Tracking</div>
        </div>
        """, unsafe_allow_html=True)
        
        costs = metrics.get("costs")
        if costs:
            st.metric("Total Spend", f"${costs.get('total_usd', 0):.4f}")
            
            by_model = costs.get("by_model", {})
            if by_model:
                st.markdown("**By Model:**")
                for model, cost in by_model.items():
                    st.markdown(f"- `{model}`: ${cost:.4f}")
        else:
            st.info("No cost data yet")
    
    # Request breakdown
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="dashboard-card">
        <div class="dashboard-title">Request Breakdown</div>
    </div>
    """, unsafe_allow_html=True)
    
    requests_by_endpoint = metrics.get("requests_by_endpoint", {})
    if requests_by_endpoint:
        # Create a simple bar display
        total = sum(requests_by_endpoint.values()) or 1
        for endpoint, count in sorted(requests_by_endpoint.items(), key=lambda x: -x[1]):
            pct = (count / total) * 100
            st.markdown(f"**{endpoint}**: {count} requests ({pct:.1f}%)")
            st.progress(pct / 100)
    else:
        st.info("No endpoint data yet")
    
    # Refresh button
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Refresh Metrics"):
        st.rerun()


# ============ Main App ============

def main():
    """Main application."""
    
    # Header
    st.markdown('<h1 class="main-header">RAG System</h1>', unsafe_allow_html=True)
    
    # Check API health
    health = check_api_health()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Configuration")
        
        # API Status
        if health:
            status_text = "[Online]" if health["status"] == "healthy" else "[Degraded]"
            st.markdown(f"{status_text} **API:** {health['status']}")
            st.caption(f"v{health['version']} - {health['environment']}")
        else:
            st.markdown("[Offline] **API:** Offline")
            st.caption("Run: `uvicorn app.main:app`")
        
        st.divider()
        
        # Workspace selection
        workspaces = get_workspaces() if health else []
        
        if workspaces:
            workspace_id = st.selectbox(
                "Workspace",
                options=workspaces,
                index=0,
                help="Select a workspace to query",
            )
        else:
            st.markdown("**Workspace**")
            st.caption("No workspaces found. Upload documents to create one.")
            workspace_id = None
        
        # Retrieval settings
        top_k = st.slider("Top K Results", 1, 10, 5)
        
        st.divider()
        
        # Quick stats
        if health:
            metrics = get_metrics()
            if metrics:
                st.markdown("### Quick Stats")
                c1, c2 = st.columns(2)
                c1.metric("Docs", metrics.get("documents_ingested", 0))
                c2.metric("Chunks", metrics.get("chunks_indexed", 0))
    
    # Main navigation
    page = st.radio(
        "Navigation",
        ["Chat", "Upload", "Dashboard"],
        horizontal=True,
        label_visibility="collapsed",
    )
    
    st.markdown("---")
    
    # ============ Chat Page ============
    if page == "Chat":
        # Initialize state
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "last_response" not in st.session_state:
            st.session_state.last_response = None
        if "show_full_citations" not in st.session_state:
            st.session_state.show_full_citations = False
        
        # Two-column layout: Chat (70%) | Evidence (30%)
        chat_col, evidence_col = st.columns([7, 3])
        
        with chat_col:
            st.markdown("### Chat")
            
            # Chat history container
            chat_container = st.container()
            with chat_container:
                if not st.session_state.messages:
                    st.markdown("""
                    <div style="text-align: center; padding: 2rem; color: var(--text-secondary);">
                        <p>Ask a question about your documents.</p>
                        <p style="font-size: 0.85rem; opacity: 0.7;">
                            Results will appear here with evidence in the right panel.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    for msg in st.session_state.messages:
                        render_chat_message(msg)
            
            # Input area
            st.markdown("---")
            query = st.text_input(
                "Ask a question",
                placeholder="What would you like to know?",
                key="query_input",
                label_visibility="collapsed",
            )
            
            btn_col1, btn_col2, _ = st.columns([1, 1, 3])
            with btn_col1:
                send = st.button("Send", use_container_width=True)
            with btn_col2:
                if st.button("Clear", use_container_width=True):
                    st.session_state.messages = []
                    st.session_state.last_response = None
                    st.rerun()
            
            # Handle send
            if send and query:
                if not health:
                    st.error("API is offline. Start the backend with: make backend")
                elif not workspace_id:
                    st.error("No workspace selected. Upload documents first.")
                else:
                    # Add user message
                    st.session_state.messages.append({"role": "user", "content": query})
                    
                    # Query API
                    with st.spinner("Processing query..."):
                        response = query_rag(workspace_id, query, top_k)
                    
                    if response:
                        # Store full response for evidence panel
                        st.session_state.last_response = response
                        
                        # Add assistant message to chat history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response.get("answer"),
                            "sources": response.get("sources", []),
                            "model": response.get("model_used"),
                            "latency": response.get("latency_ms"),
                            "retrieval_only": response.get("retrieval_only", False),
                            "refused": response.get("refused", False),
                            "refusal_reason": response.get("refusal_reason"),
                            "confidence": response.get("confidence", {}),
                            "router_decision": response.get("router_decision", {}),
                            "cost_usd": response.get("cost_usd"),
                        })
                        st.rerun()
        
        with evidence_col:
            # Evidence panel header with toggle
            st.markdown("---")  # Visual separator
            
            # Show full citations toggle
            show_full = st.checkbox(
                "Show full citations",
                value=st.session_state.show_full_citations,
                key="full_citations_toggle",
                help="Show all retrieved sources instead of top 3"
            )
            st.session_state.show_full_citations = show_full
            
            # Render evidence panel
            render_evidence_panel(
                st.session_state.last_response,
                show_full_citations=st.session_state.show_full_citations
            )
    
    # ============ Upload Page ============
    elif page == "Upload":
        st.markdown("### Upload Documents")
        
        # Workspace selection for upload (can create new or use existing)
        upload_col1, upload_col2 = st.columns([2, 1])
        
        with upload_col1:
            # Allow selecting existing or typing new workspace
            existing_workspaces = workspaces if health else []
            
            if existing_workspaces:
                use_existing = st.checkbox("Use existing workspace", value=True)
                if use_existing:
                    upload_workspace_id = st.selectbox(
                        "Select workspace",
                        options=existing_workspaces,
                        key="upload_workspace_select",
                    )
                else:
                    upload_workspace_id = st.text_input(
                        "New workspace ID",
                        value="",
                        placeholder="Enter new workspace name",
                        key="upload_workspace_new",
                    )
            else:
                upload_workspace_id = st.text_input(
                    "Workspace ID",
                    value="default",
                    placeholder="Enter workspace name",
                    key="upload_workspace_input",
                )
        
        with upload_col2:
            st.markdown("**Supported:**")
            st.markdown("- PDF, TXT, MD")
            st.markdown("- DOCX, HTML")
        
        st.markdown("---")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=["pdf", "txt", "md", "docx", "html"],
        )
        
        if uploaded_files:
            st.markdown(f"**{len(uploaded_files)} file(s) selected**")
            
            if st.button("Ingest All", use_container_width=True):
                if not health:
                    st.error("API is offline")
                elif not upload_workspace_id or not upload_workspace_id.strip():
                    st.error("Please enter a workspace ID")
                else:
                    progress = st.progress(0)
                    
                    for i, file in enumerate(uploaded_files):
                        with st.spinner(f"Processing {file.name}..."):
                            result = ingest_document(upload_workspace_id.strip(), file)
                            
                            if result:
                                st.success(
                                    f"[OK] **{file.name}** - "
                                    f"{result['chunks_created']} chunks "
                                    f"({result['processing_time_ms']:.0f}ms)"
                                )
                            else:
                                st.error(f"[Failed] {file.name}")
                        
                        progress.progress((i + 1) / len(uploaded_files))
                    
                    # Hint to refresh to see new workspace
                    st.info("Refresh the page to see new workspaces in the sidebar.")
    
    # ============ Dashboard Page ============
    elif page == "Dashboard":
        if not health:
            st.warning("Connect to API to view dashboard")
        else:
            metrics = get_metrics()
            if metrics:
                render_dashboard(metrics)
            else:
                st.info("No metrics available yet. Start using the system to generate data.")


if __name__ == "__main__":
    main()
