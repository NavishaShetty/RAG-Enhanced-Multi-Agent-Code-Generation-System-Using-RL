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
    page_icon="🤖",
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
    st.sidebar.title("🤖 Multi-Agent Code Generator")
    st.sidebar.caption("RAG-Enhanced with RL Orchestration")

    st.sidebar.divider()

    # Architecture diagram
    st.sidebar.subheader("📐 Architecture")
    st.sidebar.markdown("""
    ```
    User Input
        ↓
    RL Orchestrator
        ↓
    ┌─────────────┐
    │   RAG       │←── Knowledge Base
    └─────────────┘
        ↓
    ┌─────────────────────────┐
    │  Planner → Coder →      │
    │  Tester → Debugger      │
    │     (Blackboard)        │
    └─────────────────────────┘
        ↓
    Generated Code
    ```
    """)

    st.sidebar.divider()

    # Project info
    st.sidebar.subheader("ℹ️ About")
    st.sidebar.markdown("""
    This system uses **Reinforcement Learning** to orchestrate
    multiple LLM agents for code generation tasks.

    **Features:**
    - 🧠 Q-Learning + Thompson Sampling
    - 📚 RAG-enhanced code generation
    - 🔄 4 specialized agents
    - ⚡ Learned efficient strategies
    """)

    st.sidebar.divider()

    # System status
    st.sidebar.subheader("📊 System Status")

    if st.session_state.pipeline:
        rag_status = "✅ Active" if st.session_state.pipeline.is_rag_available() else "⚠️ Not initialized"
        rl_status = "✅ Active" if st.session_state.pipeline.is_rl_available() else "⚠️ Not available"
    else:
        rag_status = "⏳ Not initialized"
        rl_status = "⏳ Not initialized"

    st.sidebar.metric("RAG System", rag_status)
    st.sidebar.metric("RL Policy", rl_status)

    st.sidebar.divider()

    # Links
    st.sidebar.subheader("🔗 Links")
    st.sidebar.markdown("[📁 GitHub Repository](https://github.com/)")
    st.sidebar.markdown("[📄 Documentation](https://)")


def render_header():
    """Render page header."""
    st.title("🤖 Multi-Agent Code Generator")
    st.caption("Generate Python code using RL-orchestrated LLM agents with RAG enhancement")


def render_task_input():
    """Render task input section."""
    st.subheader("📝 Task Description")

    # Example tasks for quick selection
    example_tasks = [
        "Write a function that reverses a string",
        "Write a function that finds the maximum value in a list",
        "Write a function that checks if a number is prime",
        "Write a function that calculates factorial recursively",
        "Write a function that checks if a string is a palindrome",
        "Write a function that finds duplicates in a list",
        "Write a function to perform binary search",
        "Write a function that generates Fibonacci numbers",
    ]

    # Quick example buttons
    st.caption("Quick examples (click to use):")
    cols = st.columns(4)
    for i, task in enumerate(example_tasks[:4]):
        with cols[i]:
            if st.button(f"Ex {i+1}", help=task, key=f"ex_{i}"):
                st.session_state.task_input = task
                st.rerun()

    cols2 = st.columns(4)
    for i, task in enumerate(example_tasks[4:8]):
        with cols2[i]:
            if st.button(f"Ex {i+5}", help=task, key=f"ex_{i+4}"):
                st.session_state.task_input = task
                st.rerun()

    # Main input
    task = st.text_area(
        "Describe the code you want to generate:",
        value=st.session_state.task_input,
        height=100,
        placeholder="e.g., Write a function that reverses a string",
        key="task_text_area"
    )

    # Settings
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
    if st.button("🚀 Generate Code", type="primary", use_container_width=True):
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

    # Initialize pipeline
    if st.session_state.pipeline is None:
        with st.spinner("Initializing system..."):
            st.session_state.pipeline = UIPipeline(
                use_rag=use_rag,
                use_rl=use_rl,
                max_iterations=max_iter
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
        st.info("🔄 Generating code...")

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


def render_agent_activity():
    """Render real-time agent activity visualization."""
    st.subheader("🔄 Agent Activity")

    if not st.session_state.agent_logs and st.session_state.result is None:
        st.info("Submit a task to see agent activity")
        return

    # Agent status indicators
    agents = ['Planner', 'Coder', 'Tester', 'Debugger']
    agent_icons = {'Planner': '📋', 'Coder': '💻', 'Tester': '🧪', 'Debugger': '🔧'}
    agent_colors = {
        'Planner': '#FF6B6B',
        'Coder': '#4ECDC4',
        'Tester': '#45B7D1',
        'Debugger': '#96CEB4'
    }

    cols = st.columns(4)
    for col, agent in zip(cols, agents):
        with col:
            is_active = st.session_state.active_agent == agent
            status = "🟢 Active" if is_active else "⚪ Idle"

            # Check if this agent was used
            agent_used = any(log['agent'] == agent for log in st.session_state.agent_logs)
            if agent_used and not is_active:
                status = "✅ Done"

            st.metric(
                label=f"{agent_icons[agent]} {agent}",
                value=status
            )

    # Activity log
    st.caption("Activity Log")
    log_container = st.container()
    with log_container:
        for log in st.session_state.agent_logs:
            agent_name = log['agent']
            icon = agent_icons.get(agent_name, '📌')

            with st.expander(f"{icon} **{agent_name}**: {log['message']}", expanded=False):
                if log.get('details'):
                    st.code(log['details'][:1000], language="python" if agent_name == "Coder" else "text")


def render_output():
    """Render code output section."""
    st.subheader("📤 Generated Output")

    if st.session_state.generated_code is None:
        st.info("Generated code will appear here")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        # Code output with syntax highlighting
        st.code(st.session_state.generated_code, language="python")

        # Download button
        st.download_button(
            label="📥 Download Code",
            data=st.session_state.generated_code,
            file_name="generated_code.py",
            mime="text/plain"
        )

    with col2:
        # Metrics
        st.caption("📊 Metrics")
        metrics = st.session_state.metrics

        # Success status
        if metrics.get('success'):
            st.success("✅ Success")
        else:
            st.warning("⚠️ Max iterations reached")

        st.metric("Steps", metrics.get('steps', 'N/A'))
        st.metric("Time", f"{metrics.get('time', 0):.2f}s")

        # Test results
        tests = metrics.get('tests', {})
        if tests:
            st.caption("Test Results")
            passed = tests.get('passed', 0)
            total = tests.get('total', 0)
            st.metric("Tests Passed", f"{passed}/{total}")

        # Complexity metrics
        complexity = metrics.get('complexity', {})
        if complexity:
            st.caption("Code Complexity")
            st.metric("Cyclomatic", complexity.get('cyclomatic', 'N/A'))
            st.metric("Lines of Code", complexity.get('loc', 'N/A'))
            st.metric("Rating", complexity.get('rating', 'N/A'))

        # RAG context
        if st.session_state.rag_context:
            with st.expander("📚 RAG Context Used"):
                st.markdown(st.session_state.rag_context[:2000])


def main():
    """Main application entry point."""
    init_session_state()

    # Sidebar
    render_sidebar()

    # Main content
    render_header()

    st.divider()

    render_task_input()

    st.divider()

    # Two columns for activity and output
    col_activity, col_output = st.columns([1, 1])

    with col_activity:
        render_agent_activity()

    with col_output:
        render_output()


if __name__ == "__main__":
    main()
