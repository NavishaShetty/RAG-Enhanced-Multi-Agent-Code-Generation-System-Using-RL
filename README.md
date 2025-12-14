# RAG-Enhanced Multi-Agent Code Generation System Using RL

<div align="center">
  <img src="https://img.shields.io/badge/Multi--Agent%20System-4%20Agents-purple" alt="Multi-Agent System" />
  <img src="https://img.shields.io/badge/Prompt%20Engineering-LLM-4ECDC4" alt="Prompt Engineering" />
  <img src="https://img.shields.io/badge/RAG-ChromaDB-orange" alt="RAG" />
  <img src="https://img.shields.io/badge/RL-Q--Learning%20%2B%20Thompson%20Sampling-FF6B6B" alt="RL" />
  <img src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit" />
</div>

A Reinforcement Learning-based orchestrator for a multi-agent code generation system, enhanced with Retrieval-Augmented Generation (RAG) for improved code quality. The RL agent learns to optimally coordinate four specialized LLM agents (Planner, Coder, Tester, Debugger) to solve coding tasks efficiently.

## Core Concept

- **LLM agents** (via OpenRouter) handle the actual coding tasks
- **RL orchestrator** (Q-Learning + Thompson Sampling) learns WHICH agent to invoke and WHEN
- **RAG system** retrieves relevant Python patterns and examples to enhance code generation for Coder Agent
- The LLM weights are NEVER updated - only the RL policy weights are trained

### RAG-Enhanced Code Generation
The system includes a Retrieval-Augmented Generation (RAG) module that:
- Maintains a knowledge base of Python best practices and patterns
- Retrieves relevant context for each coding task
- Injects context into the Coder agent's prompt for better code quality

### Streamlit Web Interface -> **[Try it live](https://multi-agent-code-generation.streamlit.app/)**
Interactive web interface featuring:
- Real-time agent activity visualization
- Code output with syntax highlighting
- Complexity metrics dashboard
- RAG context inspection

## System Architecture

<img width="944" height="1318" alt="image" src="https://github.com/user-attachments/assets/fe7b565a-532f-40d9-8d91-888f363c93df" />

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
git clone https://github.com/NavishaShetty/RAG-Enhanced-Multi-Agent-Code-Generation-System-Using-RL.git
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

## Project Structure

```
RAG-Enhanced-Multi-Agent-Code-Generation-System-Using-RL/
├── agents/                         # LLM-powered agents
│   ├── base_agent.py               # Abstract base class
│   ├── planner_agent.py            # Task planning
│   ├── coder_agent.py              # Code generation (RAG-enhanced)
│   ├── tester_agent.py             # Code analysis
│   └── debugger_agent.py           # Bug fixing
│
├── rag/                            # RAG System
│   ├── config.py                   # RAG configuration
│   ├── knowledge_base.py           # Document loading
│   ├── vector_store.py             # ChromaDB wrapper
│   └── retriever.py                # Retrieval pipeline
│
├── knowledge_base/                 # Knowledge Base Content
│   ├── python_basics.md            # Python fundamentals
│   ├── python_patterns.md          # Design patterns
│   ├── python_stdlib.md            # Standard library
│   ├── best_practices.md           # Coding best practices
│   └── error_handling.md           # Exception patterns
│
├── ui/                             # Streamlit UI
│   └── pipeline.py                 # UI-system connector
│
├── communication/                  # Agent communication
│   └── blackboard.py               # Shared message board
│
├── tools/                          # Tools for agents
│   ├── code_executor.py            # Safe code execution
│   ├── test_runner.py              # Test generation and running
│   └── complexity_analyzer.py      # Custom complexity metrics tool
│
├── rl/                             # Reinforcement Learning
│   ├── q_learning.py               # Q-Learning implementation
│   ├── thompson_sampling.py        # Thompson Sampling
│   └── combined_agent.py           # Q-Learning + Thompson Sampling
│
├── environment/                    # RL Environment
│   ├── state.py                    # State representation
│   ├── rewards.py                  # Reward function
│   ├── simulated_env.py            # Fast simulation for training
│   └── coding_env.py               # Real LLM environment
│
├── orchestrator/                   # Pipeline orchestration
│   └── fixed_pipeline.py           # Fixed agent sequence pipeline
│
├── utils/                          # Utility modules
│   └── api.py                      # OpenRouter API client
│
├── training/                       # Training scripts
│   ├── train_simulated.py          # Train on simulation
│   ├── validate_real.py            # Validate with real LLM
│   └── evaluate.py                 # Compare policies
│
├── visualization/                  # Plotting
│   ├── learning_curves.py          # Training progress plots
│   └── q_table_viz.py              # Q-value heatmap
│
├── demo/                           # Demo scripts
│   └── demo.py                     # Command line demo
│
├── experiments/                    # Experiment outputs
│   └── results/                    # Saved Q-tables, metrics
│
├── scripts/                        # Utility scripts
│   ├── sanity_check.py             # API verification
│   ├── collect_baseline.py         # Baseline metrics collection
│   └── init_rag.py                 # RAG initialization
│
├── tests/                          # Unit tests
│   ├── test_blackboard.py          # Communication tests
│   ├── test_tools.py               # Tools tests
│   ├── test_environment.py         # Environment tests
│   ├── test_rl.py                  # RL agents tests
│   └── test_rag.py                 # RAG system tests
│
├── docs/                           # Documentation
│   ├── Technical Report 
│   ├── Summary       
│   └── Architecture Diagram     
│
├── app.py                          # Streamlit entry point
├── config.yaml                     # Configuration file
├── requirements.txt                # Python dependencies
└── README.md                       # This file
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
| Strategy | planner->coder->tester->debugger | coder->tester |

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
