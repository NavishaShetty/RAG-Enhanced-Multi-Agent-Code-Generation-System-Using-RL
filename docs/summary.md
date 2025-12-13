# Project Summary

## RAG-Enhanced Multi-Agent Code Generation System Using Reinforcement Learning

---

## What This Project Does

### The Core Problem

When you have multiple specialized LLM agents (Planner, Coder, Tester, Debugger), a fundamental question arises:

**How do you decide *which agent to call* and *when*?**

A naive approach uses a fixed pipeline:

```
Planner -> Coder -> Tester -> Debugger -> repeat
```

But this is wasteful as we don't really need to plan for a simple task like "write a function that reverses a string"

### The Solution

Instead of hardcoding orchestration logic, this project uses **Reinforcement Learning** to *learn* the optimal coordination strategy.

The RL agent observes the current workflow state and decides which agent to invoke next—learning through trial and error what works best.

---

## The Key Insight

The RL agent **discovered something non-obvious** during training:

| Approach | Strategy | Agent Calls (5 tasks) |
|----------|----------|----------------------|
| Fixed Pipeline | planner -> coder -> tester -> debugger | 23 calls |
| **Learned Policy** | coder -> tester (skip planning!) | **10 calls** |

### Result

- **56% fewer agent calls**
- **Same 100% success rate**

The RL agent learned *on its own* that for simple coding tasks, **planning is unnecessary overhead**. It discovered that going directly to coding and testing is more efficient—something a human-designed pipeline would likely miss.

### Why This Matters

This demonstrates that RL can:
- Discover non-obvious optimizations in multi-agent workflows
- Adapt orchestration strategies based on task characteristics
- Potentially outperform human-designed pipelines

This has applications beyond code generation—any system where multiple AI agents need coordination (customer service, research assistants, automated workflows) could benefit from learned orchestration.

---

## The RAG Enhancement

To make the Coder agent smarter and more reliable, the system incorporates **Retrieval-Augmented Generation (RAG)**:

### 1. Knowledge Base
Five curated markdown files containing:
- Python fundamentals and common patterns
- Standard library reference
- Best practices and coding conventions
- Error handling patterns

### 2. Vector Store
- **ChromaDB** for persistent vector storage
- **Sentence-transformers** (all-MiniLM-L6-v2) for embeddings
- Semantic search to find relevant content

### 3. Retrieval Pipeline
When the Coder agent is invoked:
1. The task description is used as a query
2. Relevant code examples and patterns are retrieved
3. Context is injected into the Coder's prompt

### Before vs After RAG

**Without RAG:**
```
Prompt: "Write a function that reverses a string"
```

**With RAG:**
```
Prompt: "Write a function that reverses a string"

Relevant Context from Knowledge Base:
- Python string slicing: s[::-1] reverses a string
- Best practice: handle edge cases (empty string, None)
- Example: def reverse_string(s): return s[::-1] if s else ""
```

This gives the LLM concrete examples and best practices, improving code quality and consistency.

---

## System Architecture

```
                        ┌─────────────────────┐
                        │    User Input       │
                        │  (Task Description) │
                        └──────────┬──────────┘
                                   │
                        ┌──────────▼──────────┐
                        │   Streamlit UI      │
                        └──────────┬──────────┘
                                   │
           ┌───────────────────────▼───────────────────────┐
           │              RL ORCHESTRATOR                  │
           │         Q-Learning + Thompson Sampling        │
           │                                               │
           │  State: (has_plan, has_code, has_error,       │
           │          tests_pass, iteration)               │
           │                                               │
           │  Action: Which agent to call next?            │
           └───────────────────────┬───────────────────────┘
                                   │
          ┌────────────┬───────────┼───────────┬────────────┐
          ▼            ▼           ▼           ▼            │
     ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐        │
     │ Planner │ │  Coder  │ │ Tester  │ │Debugger │        │
     │         │ │         │ │         │ │         │        │
     │ Breaks  │ │ Writes  │ │  Runs   │ │  Fixes  │        │
     │  down   │ │  code   │ │  tests  │ │  bugs   │        │
     │  tasks  │ │         │ │         │ │         │        │
     └─────────┘ └────┬────┘ └─────────┘ └─────────┘        │
                      │                                     │
                      ▼                                     │
               ┌─────────────┐                              │
               │ RAG System  │                              │
               │             │                              │
               │ ChromaDB +  │                              │
               │ Embeddings  │                              │
               └──────┬──────┘                              │
                      │                                     │
                      ▼                                     │
               ┌─────────────┐                              │
               │ Knowledge   │                              │
               │    Base     │                              │
               │  (5 files)  │                              │
               └─────────────┘                              │
                                                            │
                       ┌────────────────────────────────────┘
                       │
                       ▼
    ┌───────────────────────────────────────┐
    │            BLACKBOARD                 │
    │      (Shared Communication)           │
    │                                       │
    │  Messages: task, plan, code,          │
    │            test_result, error         │
    └──────────────────┬────────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Generated Code │
              │  + Test Results │
              │  + Metrics      │
              └─────────────────┘
```

---

## Technical Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| RL Algorithm | Q-Learning + Thompson Sampling | Learn optimal agent orchestration |
| State Space | 64 discrete states | Compact representation of workflow status |
| Action Space | 4 actions | planner, coder, tester, debugger |
| LLM Backend | OpenRouter API | Powers all agent responses |
| RAG Vector Store | ChromaDB | Semantic search over knowledge base |
| Embeddings | all-MiniLM-L6-v2 | CPU-friendly sentence embeddings |
| Communication | Blackboard Pattern | Message passing between agents |
| UI | Streamlit | Interactive web interface |
| Custom Tool | Complexity Analyzer | Code quality metrics |

---

## Results Summary

### Training Performance
- **Training Time:** < 1 second (5,000 episodes)
- **Final Success Rate:** 97% (simulation)
- **Validation Success Rate:** 100% (real LLM)

### Policy Comparison

| Metric | Fixed Pipeline | Learned Policy | Improvement |
|--------|----------------|----------------|-------------|
| Success Rate | 100% | 100% | — |
| Total Agent Calls (5 tasks) | 23 | 10 | **-56.5%** |
| Strategy | Always plan first | Skip planning | Adaptive |

### Learned Q-Values (Initial State)

| Action | Q-Value | Interpretation |
|--------|---------|----------------|
| coder | **8.48** | Strongly preferred |
| planner | 0.04 | Rarely chosen |
| tester | 0.00 | Invalid without code |
| debugger | 0.00 | Invalid without error |

The high Q-value for "coder" in the initial state shows the agent learned to skip planning entirely for simple tasks.

---

## Key Takeaways

1. **RL Discovers Efficient Strategies:** The agent learned to skip unnecessary planning, reducing agent calls by 56%.

2. **Simulation-to-Real Transfer Works:** Training on a fast simulation environment (100k episodes/second) produced policies that work on real LLM calls.

3. **RAG Improves Code Quality:** Injecting relevant context helps the Coder agent produce better, more consistent code.

4. **Tabular Methods Are Sufficient:** With only 64 states, simple Q-learning works well—no need for deep RL.

5. **Practical Implications:** This approach can be applied to any multi-agent system where orchestration decisions matter.

---

## Future Directions

- Extend to more complex, multi-step coding tasks
- Add task complexity detection to adaptively use planning
- Explore deep RL for larger state spaces
- Integrate additional specialized agents
- Build larger, domain-specific knowledge bases

---

*This project demonstrates that Reinforcement Learning can effectively learn optimal orchestration strategies for multi-agent LLM systems, discovering non-obvious optimizations that improve efficiency without sacrificing performance.*