# RAG-Enhanced Multi-Agent Code Generation System Using RL

A Reinforcement Learning-based orchestrator for a multi-agent code generation system, enhanced with **Retrieval-Augmented Generation (RAG)** for improved code quality. The RL agent learns to optimally coordinate four specialized LLM agents (Planner, Coder, Tester, Debugger) to solve coding tasks efficiently.

## Core Concept

- **LLM agents** (via OpenRouter) handle the actual coding tasks
- **RL orchestrator** (Q-Learning + Thompson Sampling) learns WHICH agent to invoke and WHEN
- **RAG system** retrieves relevant Python patterns and examples to enhance code generation
- The LLM weights are NEVER updated - only the RL policy weights are trained

## New Features (RAG + Streamlit UI)

### RAG-Enhanced Code Generation
The system now includes a Retrieval-Augmented Generation (RAG) module that:
- Maintains a knowledge base of Python best practices and patterns
- Retrieves relevant context for each coding task
- Injects context into the Coder agent's prompt for better code quality

### Streamlit Web Interface
Interactive web interface featuring:
- Real-time agent activity visualization
- Code output with syntax highlighting
- Complexity metrics dashboard
- RAG context inspection

![Architecture](docs/architecture.png)

## Features

- **4 Specialized Agents**: Planner, Coder, Tester, Debugger
- **2 RL Methods**: Q-Learning (value-based) + Thompson Sampling (exploration)
- **RAG System**: ChromaDB + Sentence Transformers for semantic search
- **Blackboard Communication**: Shared message passing between agents
- **Custom Tool**: Code Complexity Analyzer with multiple metrics
- **Streamlit UI**: Interactive web interface for code generation
- **Fast Simulation**: ~100k episodes/second for training
- **Visualization**: Learning curves and Q-table heatmaps

## Quick Start (Updated)

### Prerequisites
- Python 3.10+
- OpenRouter API key

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd RAG-Enhanced-Multi-Agent-Code-Generation-System-Using-RL
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your OpenRouter API key:
```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

5. Initialize the RAG system:
```bash
python scripts/init_rag.py
```

6. Run the Streamlit UI:
```bash
streamlit run app.py
```

## Usage

### Option 1: Streamlit Web Interface (Recommended)
```bash
streamlit run app.py
```
Then open http://localhost:8501 in your browser.

### Option 2: Command Line Demo
```bash
python demo/demo.py
```

### Option 3: Interactive Demo
```bash
python orchestrator/fixed_pipeline.py
```

## Project Structure

```
RAG-Enhanced-Multi-Agent-Code-Generation-System-Using-RL/
├── agents/                   # LLM-powered agents
│   ├── base_agent.py         # Abstract base class
│   ├── planner_agent.py      # Task planning
│   ├── coder_agent.py        # Code generation (RAG-enhanced)
│   ├── tester_agent.py       # Code analysis
│   └── debugger_agent.py     # Bug fixing
│
├── rag/                      # RAG System (NEW)
│   ├── __init__.py
│   ├── config.py             # RAG configuration
│   ├── knowledge_base.py     # Document loading
│   ├── vector_store.py       # ChromaDB wrapper
│   └── retriever.py          # Retrieval pipeline
│
├── knowledge_base/           # Knowledge Base Content (NEW)
│   ├── python_basics.md      # Python fundamentals
│   ├── python_patterns.md    # Design patterns
│   ├── python_stdlib.md      # Standard library
│   ├── best_practices.md     # Coding best practices
│   └── error_handling.md     # Exception patterns
│
├── ui/                       # Streamlit UI (NEW)
│   ├── __init__.py
│   └── pipeline.py           # UI-system connector
│
├── communication/            # Agent communication
│   └── blackboard.py         # Shared message board
│
├── tools/                    # Tools for agents
│   ├── code_executor.py      # Safe code execution
│   ├── test_runner.py        # Test generation and running
│   └── complexity_analyzer.py    # Custom complexity metrics tool
│
├── rl/                       # Reinforcement Learning
│   ├── q_learning.py         # Q-Learning implementation
│   ├── thompson_sampling.py  # Thompson Sampling
│   └── combined_agent.py     # Q-Learning + Thompson Sampling
│
├── environment/              # RL Environment
│   ├── state.py              # State representation
│   ├── rewards.py            # Reward function
│   ├── simulated_env.py      # Fast simulation for training
│   └── coding_env.py         # Real LLM environment
│
├── orchestrator/             # Pipeline orchestration
│   └── fixed_pipeline.py     # Fixed agent sequence pipeline
│
├── training/                 # Training scripts
│   ├── train_simulated.py    # Train on simulation
│   ├── validate_real.py      # Validate with real LLM
│   └── evaluate.py           # Compare policies
│
├── visualization/            # Plotting
│   ├── learning_curves.py    # Training progress plots
│   └── q_table_viz.py        # Q-value heatmap
│
├── scripts/                  # Utility scripts
│   ├── sanity_check.py       # API verification
│   ├── collect_baseline.py   # Baseline metrics collection
│   └── init_rag.py           # RAG initialization (NEW)
│
├── tests/                    # Unit tests
│   ├── test_blackboard.py    # Communication tests
│   ├── test_tools.py         # Tools tests
│   ├── test_environment.py   # Environment tests
│   ├── test_rl.py            # RL agents tests
│   └── test_rag.py           # RAG system tests (NEW)
│
├── docs/                     # Documentation (NEW)
│   ├── index.html            # GitHub Pages website
│   └── architecture.png      # Architecture diagram
│
├── chroma_db/                # Vector store data (NEW, gitignored)
│
├── app.py                    # Streamlit entry point (NEW)
├── config.yaml               # Configuration file
├── requirements.txt          # Python dependencies (updated)
└── README.md                 # This file
```

## RAG System Details

### Knowledge Base
The knowledge base contains curated Python programming knowledge:

| File | Content | Examples |
|------|---------|----------|
| `python_basics.md` | Fundamentals | String reversal, max finding, sorting |
| `python_patterns.md` | Design patterns | Comprehensions, decorators, generators |
| `python_stdlib.md` | Standard library | collections, itertools, functools |
| `best_practices.md` | Code quality | PEP 8, type hints, DRY principle |
| `error_handling.md` | Exceptions | try/except, custom exceptions, logging |

### Retrieval Pipeline
1. User submits a coding task
2. RAG retrieves top-k relevant documents from knowledge base
3. Context is injected into the Coder agent's prompt
4. LLM generates code with enhanced context

### Configuration
RAG settings can be customized:
```python
from rag.config import RAGConfig

config = RAGConfig(
    embedding_model="all-MiniLM-L6-v2",  # CPU-friendly
    top_k=5,                               # Number of documents
    relevance_threshold=0.3,               # Minimum similarity
    chunk_size=500,                        # Chunk size in chars
)
```

## RL Formulation

### State Space (64 states)
- `has_plan`: bool (2 values)
- `has_code`: bool (2 values)
- `has_error`: bool (2 values)
- `tests_pass`: bool (2 values)
- `iteration_bucket`: 0-3 (4 values)

Total: 2 * 2 * 2 * 2 * 4 = 64 states

### Action Space (4 actions)
- `planner`: Generate task plan
- `coder`: Write code (with RAG context)
- `tester`: Analyze/test code
- `debugger`: Fix errors

### Reward Function
| Event | Reward |
|-------|--------|
| Task success | +10.0 |
| Task timeout | -5.0 |
| Progress (plan) | +0.2 |
| Progress (code) | +0.3 |
| Error fixed | +0.5 |
| Redundant action | -0.2 |
| Invalid action | -0.3 |
| Step cost | -0.1 |

## Running Tests

Run all tests:
```bash
python -m pytest tests/ -v
```

Run RAG tests specifically:
```bash
python -m pytest tests/test_rag.py -v
```

Run with coverage:
```bash
python -m pytest tests/ --cov=. --cov-report=html
```

## Training the RL Agent

### 1. Train on Simulation (Fast)
```bash
python training/train_simulated.py --episodes 5000
```

### 2. Validate with Real LLM
```bash
python training/validate_real.py --tasks 5
```

### 3. Compare Results
```bash
python training/evaluate.py
```

## Results

After training, the RL agent learns an efficient policy:

| Metric | Fixed Pipeline | Learned Policy |
|--------|---------------|----------------|
| Success Rate | 100% | 100% |
| Avg Steps | 1.8 iterations | 2 steps |
| Strategy | planner→coder→tester→debugger | coder→tester |

Key findings:
- Training completes in less than 1 second for 5000 episodes
- The agent learns to skip planning for simple tasks
- RAG improves code quality by providing relevant context

## Custom Tool: Complexity Analyzer

Located in `tools/complexity_analyzer.py`, this tool provides:
- Cyclomatic complexity
- Lines of code
- Function count
- Max nesting depth
- Cognitive complexity

```python
from tools.complexity_analyzer import ComplexityAnalyzer

analyzer = ComplexityAnalyzer()
metrics = analyzer.analyze(code)
print(f"Complexity score: {metrics['overall_score']}")
```

## Configuration

Edit `config.yaml` to customize:
- API settings (model, temperature)
- RL hyperparameters (α, γ, episodes)
- Simulation probabilities

## License

MIT License
