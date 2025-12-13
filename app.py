"""
Streamlit UI for RAG-Enhanced Multi-Agent Code Generation System.

Run with: streamlit run app.py
"""

import streamlit as st
import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.pipeline import UIPipeline, PipelineResult

# Page configuration
st.set_page_config(
    page_title="Multi-Agent Code Generator",
    page_icon="</>",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .agent-card {
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
    }
    .agent-active {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .agent-idle {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    .stCodeBlock {
        max-height: 500px;
    }
</style>
""", unsafe_allow_html=True)


def get_api_key():
    """Get API key from Streamlit secrets or environment variable."""
    try:
        return st.secrets["OPENROUTER_API_KEY"]
    except (KeyError, FileNotFoundError):
        return os.getenv("OPENROUTER_API_KEY")


@st.cache_resource
def initialize_rag_system():
    """Initialize RAG system once at app startup."""
    from rag.retriever import Retriever
    retriever = Retriever()
    retriever.initialize()
    return retriever


def init_session_state():
    """Initialize session state variables."""
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'generated_code' not in st.session_state:
        st.session_state.generated_code = None
    if 'agent_logs' not in st.session_state:
        st.session_state.agent_logs = []
    if 'active_agent' not in st.session_state:
        st.session_state.active_agent = None
    if 'metrics' not in st.session_state:
        st.session_state.metrics = {}
    if 'rag_context' not in st.session_state:
        st.session_state.rag_context = ""
    if 'is_generating' not in st.session_state:
        st.session_state.is_generating = False
    if 'task_input' not in st.session_state:
        st.session_state.task_input = ""
    if 'result' not in st.session_state:
        st.session_state.result = None


def render_sidebar():
    """Render sidebar with settings and info."""
    st.sidebar.subheader("ℹ️ About")
    st.sidebar.markdown("""
    This system uses **Reinforcement Learning** to orchestrate
    multiple LLM agents for code generation tasks.

    **Features:**
    - Q-Learning + Thompson Sampling
    - RAG-enhanced code generation
    - 4 specialized agents: Planner, Coder, Tester & Debugger
    - Learned efficient strategies
    - LLM powered (via OpenRouter)
    """)

    st.sidebar.divider()

    # Metrics section (moved from main page)
    st.sidebar.subheader("▶️ Metrics")
    if st.session_state.metrics:
        metrics = st.session_state.metrics

        # Success status
        if metrics.get('success'):
            st.sidebar.success("Success")
        else:
            st.sidebar.warning("Max iterations reached")

        st.sidebar.text(f"Steps: {metrics.get('steps', 'N/A')}")
        st.sidebar.text(f"Time: {metrics.get('time', 0):.2f}s")

        # Test results
        tests = metrics.get('tests', {})
        if tests:
            passed = tests.get('passed', 0)
            total = tests.get('total', 0)
            st.sidebar.text(f"Tests: {passed}/{total}")

        # Complexity metrics
        complexity = metrics.get('complexity', {})
        if complexity:
            st.sidebar.caption("Code Complexity")
            st.sidebar.text(f"Cyclomatic: {complexity.get('cyclomatic', 'N/A')}")
            st.sidebar.text(f"Lines: {complexity.get('loc', 'N/A')}")
            st.sidebar.text(f"Rating: {complexity.get('rating', 'N/A')}")

        # RAG context
        if st.session_state.rag_context:
            with st.sidebar.expander("RAG Context Used"):
                st.markdown(st.session_state.rag_context[:2000])
    else:
        st.sidebar.caption("Generate code to see metrics")

    st.sidebar.divider()

    # Links
    st.sidebar.subheader("▶️ Links")
    st.sidebar.markdown("[ GitHub Repository](https://github.com/NavishaShetty/RAG-Enhanced-Multi-Agent-Code-Generation-System-Using-RL)")
    st.sidebar.markdown("[ Documentation](https://github.com/NavishaShetty/RAG-Enhanced-Multi-Agent-Code-Generation-System-Using-RL/blob/main/docs/Technical%20Report.pdf)")


def render_header():
    """Render page header."""
    st.title("</> Multi-Agent Code Generator")
    st.caption("RAG-Enhanced with RL Orchestration")


def render_task_input():
    """Render task input section."""
    st.subheader("What do you need help with?")

    # Example tasks for dropdown
    example_tasks = {
        "Select an example...": "",
        "Reverse a string": "Write a function that reverses a string",
        "Find maximum in list": "Write a function that finds the maximum value in a list",
        "Check if prime": "Write a function that checks if a number is prime",
        "Calculate factorial": "Write a function that calculates factorial recursively",
    }

    def on_example_change():
        """Callback when example dropdown changes."""
        selected = st.session_state.example_dropdown
        if selected != "Select an example..." and example_tasks.get(selected):
            st.session_state.task_input = example_tasks[selected]

    # Dropdown for examples
    st.selectbox(
        "Choose an example or type your own below:",
        options=list(example_tasks.keys()),
        key="example_dropdown",
        on_change=on_example_change
    )

    # Main input
    task = st.text_area(
        "Your coding question:",
        value=st.session_state.task_input,
        height=100,
        placeholder="Type your question here...",
        label_visibility="collapsed"
    )

    # Update session state when user types
    st.session_state.task_input = task

    # Settings in expander
    with st.expander("Advanced Settings"):
        col1, col2, col3 = st.columns(3)
        with col1:
            use_rag = st.checkbox(
                "Use RAG",
                value=True,
                help="Retrieve relevant context from knowledge base"
            )
        with col2:
            use_rl = st.checkbox(
                "Use RL Policy",
                value=True,
                help="Use learned orchestration policy"
            )
        with col3:
            max_iter = st.slider(
                "Max Iterations",
                min_value=1,
                max_value=10,
                value=5,
                help="Maximum number of agent iterations"
            )

    # Generate button
    if st.button("Generate Code", type="primary", use_container_width=True):
        if task.strip():
            run_generation(task, use_rag, use_rl, max_iter)
        else:
            st.warning("Please enter a task description")


def run_generation(task: str, use_rag: bool, use_rl: bool, max_iter: int):
    """Run the code generation pipeline."""
    st.session_state.is_generating = True
    st.session_state.agent_logs = []
    st.session_state.generated_code = None
    st.session_state.metrics = {}
    st.session_state.rag_context = ""

    # Get API key
    api_key = get_api_key()

    # Initialize pipeline
    if st.session_state.pipeline is None:
        with st.spinner("Initializing system..."):
            st.session_state.pipeline = UIPipeline(
                use_rag=use_rag,
                use_rl=use_rl,
                max_iterations=max_iter,
                api_key=api_key
            )
            st.session_state.pipeline.initialize()
    else:
        # Update settings
        st.session_state.pipeline.use_rag = use_rag
        st.session_state.pipeline.use_rl = use_rl
        st.session_state.pipeline.max_iterations = max_iter

    # Progress placeholder
    progress_placeholder = st.empty()
    log_placeholder = st.empty()

    # Run pipeline
    with progress_placeholder.container():
        st.info("Generating code...")

        def on_log(entry):
            st.session_state.agent_logs.append(entry)
            st.session_state.active_agent = entry['agent']

        result = st.session_state.pipeline.run(
            task,
            on_log=on_log
        )

    # Store results
    st.session_state.result = result
    st.session_state.generated_code = result.code
    st.session_state.rag_context = result.rag_context
    st.session_state.metrics = {
        'success': result.success,
        'steps': result.steps,
        'time': result.time_seconds,
        'complexity': result.complexity_metrics,
        'tests': result.test_results
    }
    st.session_state.is_generating = False
    st.session_state.active_agent = None

    # Clear progress and rerun to show results
    progress_placeholder.empty()
    st.rerun()


def render_output():
    """Render code output section."""
    st.subheader("Generated Output")

    if st.session_state.generated_code is None:
        st.info("Generated code will appear here")
        return

    # Code output with syntax highlighting
    st.code(st.session_state.generated_code, language="python")

    # Download button
    st.download_button(
        label="Download Code",
        data=st.session_state.generated_code,
        file_name="generated_code.py",
        mime="text/plain"
    )


def render_agent_activity():
    """Render real-time agent activity visualization."""
    st.subheader("Agent Activity")

    if not st.session_state.agent_logs and st.session_state.result is None:
        st.info("Submit a task to see agent activity")
        return

    # Agent status indicators
    agents = ['Planner', 'Coder', 'Tester', 'Debugger']
    agent_icons = {'Planner': '🅿', 'Coder': '🅲', 'Tester': '🆃', 'Debugger': '🅳'}

    cols = st.columns(4)
    for col, agent in zip(cols, agents):
        with col:
            is_active = st.session_state.active_agent == agent
            status = "🟢 Active" if is_active else "⚪"

            # Check if this agent was used
            agent_used = any(log['agent'] == agent for log in st.session_state.agent_logs)
            if agent_used and not is_active:
                status = "🟢"

            # Larger font for agent label
            st.markdown(f"### {agent_icons[agent]} {agent}")
            st.markdown(f"**{status}**")

    # Activity log
    st.markdown("#### Activity Log")
    log_container = st.container()
    with log_container:
        for log in st.session_state.agent_logs:
            agent_name = log['agent']
            icon = agent_icons.get(agent_name, '▶')

            with st.expander(f"{icon} **{agent_name}**: {log['message']}", expanded=False):
                if log.get('details'):
                    st.code(log['details'][:1000], language="python" if agent_name == "Coder" else "text")


def main():
    """Main application entry point."""
    init_session_state()

    # Initialize RAG system at startup (cached - only runs once)
    with st.spinner("Initializing RAG system..."):
        initialize_rag_system()

    # Sidebar
    render_sidebar()

    # Main content
    render_header()

    st.divider()

    render_task_input()

    st.divider()

    render_output()

    st.divider()

    render_agent_activity()


if __name__ == "__main__":
    main()
