# CLAUDE.md - RL Code Generation Workflow Agent

## Project Overview

This project implements a **Reinforcement Learning-based orchestrator** for a multi-agent code generation system. The RL agent learns to optimally coordinate four specialized LLM agents (Planner, Coder, Tester, Debugger) to solve coding tasks efficiently.

### Core Concept
- **LLM agents** (via OpenRouter/Grok) handle the actual coding tasks
- **RL orchestrator** (Q-Learning + Thompson Sampling) learns WHICH agent to invoke and WHEN
- The LLM weights are NEVER updated - only the RL policy weights are trained

### Two RL Methods (Required by Assignment)
1. **Q-Learning** (Value-Based Learning) - learns value of state-action pairs
2. **Thompson Sampling** (Exploration Strategy) - balances exploration vs exploitation

### Agentic System Type
**Agentic Workflow Systems** - learning optimal planning and execution sequences

---

## Assignment Rubric Alignment

### Technical Implementation (40 points)

| Criterion | Points | How We Address It | Deliverable |
|-----------|--------|-------------------|-------------|
| Controller Design | 10 | RL Orchestrator with Q-Learning + Thompson Sampling | `orchestrator/controller.py` |
| Agent Integration | 10 | 4 specialized agents with Blackboard communication | `agents/*.py`, `communication/blackboard.py` |
| Tool Implementation | 10 | Code executor, test runner, linter | `tools/*.py` |
| Custom Tool Development | 10 | Complexity Analyzer (cyclomatic, cognitive, nesting) | `tools/complexity_analyzer.py` |

### Results and Analysis (30 points)

| Criterion | Points | How We Address It | Deliverable |
|-----------|--------|-------------------|-------------|
| Learning Performance | 15 | Learning curves, convergence plots, before/after comparison | `experiments/results/` |
| Analysis Depth | 15 | Q-table inspection, policy interpretation, failure analysis | Report Section 5 |

### Documentation and Presentation (10 points)

| Criterion | Points | How We Address It | Deliverable |
|-----------|--------|-------------------|-------------|
| Technical Documentation | 5 | README, docstrings, architecture diagram | `README.md`, `docs/` |
| Presentation Quality | 5 | 10-min video, visualizations | `demo/video.mp4` |

### Quality/Portfolio Score (20 points)

| Target | How We Achieve It |
|--------|-------------------|
| Real-world relevance | Tool selection is a real problem in LLM orchestration |
| Technical sophistication | Two RL methods working together, principled exploration |
| Innovation | Thompson Sampling for agent selection uncertainty |
| Polish | Clean code, clear visualizations, reproducible experiments |

---

## Project Structure

```
code-gen-rl/
├── CLAUDE.md                 # THIS FILE - project instructions
├── README.md                 # Setup and usage instructions
├── requirements.txt          # Python dependencies
├── config.yaml               # Configuration (API keys, hyperparameters)
│
├── agents/                   # LLM-powered agents
│   ├── __init__.py
│   ├── base_agent.py         # Abstract base class with LLM calling
│   ├── planner_agent.py      # Breaks down coding tasks
│   ├── coder_agent.py        # Writes Python code
│   ├── tester_agent.py       # Analyzes code for issues
│   └── debugger_agent.py     # Fixes errors in code
│
├── communication/            # Agent communication
│   ├── __init__.py
│   └── blackboard.py         # Shared message passing system
│
├── tools/                    # Tools available to agents
│   ├── __init__.py
│   ├── code_executor.py      # Safely runs Python code
│   ├── test_runner.py        # Runs and evaluates tests
│   ├── linter.py             # Static analysis (optional)
│   └── complexity_analyzer.py # CUSTOM TOOL - code complexity metrics
│
├── rl/                       # Reinforcement Learning
│   ├── __init__.py
│   ├── q_learning.py         # Tabular Q-Learning implementation
│   ├── thompson_sampling.py  # Thompson Sampling implementation
│   └── combined_agent.py     # Q-Learning + Thompson Sampling
│
├── environment/              # RL Environment
│   ├── __init__.py
│   ├── state.py              # State representation
│   ├── rewards.py            # Reward function definitions
│   ├── coding_env.py         # Real environment (uses LLM)
│   └── simulated_env.py      # Fast simulated environment
│
├── training/                 # Training scripts
│   ├── train_simulated.py    # Train on simulation
│   ├── validate_real.py      # Validate with real LLM
│   └── evaluate.py           # Evaluation utilities
│
├── experiments/              # Experiment tracking
│   ├── configs/              # Experiment configs
│   └── results/              # Saved results, learning curves
│
├── visualization/            # Plotting and visualization
│   ├── learning_curves.py    # Plot training progress
│   └── q_table_viz.py        # Visualize learned Q-values
│
├── demo/                     # Demonstration
│   ├── demo.py               # Interactive demo script
│   └── run_comparison.py     # Before/after comparison
│
└── tests/                    # Unit tests
    ├── test_agents.py
    ├── test_rl.py
    └── test_environment.py
```

---

## Version Checkpoints

Complete these in order. Each version should be fully working before moving to the next.

---

## V0: Sanity Check (1 hour)

### Goal
Verify that OpenRouter API works and the LLM can produce working code.

### Tasks

- [ ] **V0.1: Setup OpenRouter Connection**
  - Create `config.yaml` with API key placeholder
  - Create `utils/api.py` with OpenRouter client
  - Test basic API call returns a response
  - **Success**: Get any response from Grok

- [ ] **V0.2: Test Code Generation**
  - Create `scripts/sanity_check.py`
  - Test 5 simple coding tasks:
    1. "Write a function that returns the sum of two numbers"
    2. "Write a function that reverses a string"
    3. "Write a function that checks if a number is even"
    4. "Write a function that finds the max in a list"
    5. "Write a function that counts vowels in a string"
  - Manually verify outputs
  - Record success/failure for each
  - **Success**: At least 3/5 produce syntactically valid Python

### Deliverables
- `config.yaml` (with placeholder for API key)
- `utils/api.py`
- `scripts/sanity_check.py`
- Console output showing LLM responses

### Exit Criteria
- [x] API connection works
- [x] LLM produces Python code
- [x] At least 50% of simple tasks produce valid code

---

## V1: Multi-Agent Pipeline with Fixed Orchestration (3 hours)

### Goal
Build complete multi-agent system with hardcoded orchestration (Planner → Coder → Tester → Debugger loop).

### Tasks

- [ ] **V1.1: Blackboard Communication System**
  - Create `communication/blackboard.py`
  - Implement `Message` dataclass with: sender, receiver, content, message_type, timestamp
  - Implement `Blackboard` class with: post(), get_messages_for(), get_latest_by_type()
  - Message types: "task", "plan", "code", "test_result", "error", "feedback"
  - **Success**: Can post and retrieve messages

- [ ] **V1.2: Base Agent Class**
  - Create `agents/base_agent.py`
  - Abstract base class with: `__init__(llm_endpoint, model_name)`, `call_llm(prompt)`, abstract `_get_system_prompt()`
  - Handle API errors gracefully
  - **Success**: Base class can make LLM calls

- [ ] **V1.3: Implement Four Agents**
  - Create `agents/planner_agent.py`:
    - System prompt: "You are a planning agent. Break down coding tasks into clear steps."
    - Method: `generate_plan(task, blackboard) -> str`
    - Posts plan to blackboard with type "plan"
  - Create `agents/coder_agent.py`:
    - System prompt: "You are a coding agent. Write clean Python code. Output only code in ```python blocks."
    - Method: `generate_code(task, blackboard) -> str`
    - Reads plan from blackboard, posts code with type "code"
  - Create `agents/tester_agent.py`:
    - System prompt: "You are a testing agent. Analyze code and identify issues."
    - Method: `analyze_code(code, blackboard) -> str`
    - Posts analysis with type "feedback"
  - Create `agents/debugger_agent.py`:
    - System prompt: "You are a debugging agent. Fix code errors. Output only corrected code."
    - Method: `fix_code(code, error, blackboard) -> str`
    - Reads error from blackboard, posts fixed code
  - **Success**: Each agent can be called and produces output

- [ ] **V1.4: Code Executor Tool**
  - Create `tools/code_executor.py`
  - Safely execute Python code with timeout (5 seconds)
  - Capture stdout, stderr, exceptions
  - Return: `(success: bool, output: str, error: str)`
  - Use subprocess or restricted exec
  - **Success**: Can run simple Python and catch errors

- [ ] **V1.5: Test Runner Tool**
  - Create `tools/test_runner.py`
  - Generate simple test cases for a function
  - Run code with test inputs
  - Return: `(tests_passed: int, tests_failed: int, error_messages: list)`
  - **Success**: Can test a simple function

- [ ] **V1.6: Custom Tool - Complexity Analyzer**
  - Create `tools/complexity_analyzer.py`
  - Use Python `ast` module to analyze code
  - Metrics:
    - `cyclomatic_complexity`: count of decision points (if, for, while, except) + 1
    - `lines_of_code`: non-empty lines
    - `num_functions`: count of function definitions
    - `max_nesting_depth`: deepest nesting level
    - `cognitive_complexity`: weighted complexity score
  - Return `ComplexityMetrics` dataclass
  - Method: `overall_score() -> float` (lower is better)
  - **Success**: Can analyze and score Python code

- [ ] **V1.7: Fixed Orchestration Pipeline**
  - Create `orchestrator/fixed_pipeline.py`
  - Implement fixed sequence: Planner → Coder → Tester → (Debugger → Tester)* 
  - Max iterations: 5
  - Stop when tests pass or max iterations reached
  - Track metrics: success, iterations, agent calls
  - **Success**: Can run end-to-end on a coding task

- [ ] **V1.8: Baseline Metrics Collection**
  - Create `scripts/collect_baseline.py`
  - Run fixed pipeline on 10 coding tasks
  - Record for each: success/fail, iterations, time, complexity score
  - Save results to `experiments/results/baseline.json`
  - **Success**: Have baseline numbers for comparison

### Deliverables
- All agent files in `agents/`
- `communication/blackboard.py`
- All tool files in `tools/`
- `orchestrator/fixed_pipeline.py`
- `experiments/results/baseline.json`

### Rubric Alignment
- **Agent Integration (10 pts)**: 4 agents with defined roles ✓
- **Communication protocols**: Blackboard system ✓
- **Tool Implementation (10 pts)**: Executor, test runner ✓
- **Custom Tool (10 pts)**: Complexity analyzer ✓

### Exit Criteria
- [ ] All 4 agents work independently
- [ ] Blackboard communication works
- [ ] Code executor safely runs Python
- [ ] Complexity analyzer produces metrics
- [ ] Fixed pipeline completes tasks
- [ ] Baseline metrics collected

---

## V2: RL Implementation (3 hours)

### Goal
Implement Q-Learning + Thompson Sampling and train on simulated environment.

### Tasks

- [ ] **V2.1: State Representation**
  - Create `environment/state.py`
  - State features (keep simple for tabular RL):
    ```python
    state = {
        "has_plan": bool,        # 2 values
        "has_code": bool,        # 2 values
        "has_error": bool,       # 2 values
        "tests_pass": bool,      # 2 values
        "iteration_bucket": int  # 0, 1, 2, 3+ (4 values)
    }
    # Total: 2 × 2 × 2 × 2 × 4 = 64 states
    ```
  - Implement `to_features() -> dict` method
  - Implement `to_key() -> tuple` for Q-table lookup
  - **Success**: Can represent and hash states

- [ ] **V2.2: Action Space**
  - Actions are which agent to invoke:
    ```python
    actions = ["planner", "coder", "tester", "debugger"]
    # 4 actions
    ```
  - Total Q-table size: 64 states × 4 actions = 256 entries
  - **Success**: Action space defined

- [ ] **V2.3: Reward Function**
  - Create `environment/rewards.py`
  - Reward structure:
    ```python
    REWARDS = {
        "task_success": +10.0,      # Tests pass - big reward
        "task_timeout": -5.0,       # Max iterations exceeded
        "progress_plan": +0.2,      # Made a plan
        "progress_code": +0.3,      # Wrote code
        "tests_pass_partial": +1.0, # Some tests pass
        "error_fixed": +0.5,        # Debugger fixed an error
        "redundant_action": -0.5,   # Called planner when already have plan
        "invalid_action": -0.3,     # Called tester without code
        "step_cost": -0.1,          # Small cost per step (encourages efficiency)
    }
    ```
  - Implement `calculate_reward(state, action, next_state, done) -> float`
  - **Success**: Reward function implemented

- [ ] **V2.4: Simulated Environment**
  - Create `environment/simulated_env.py`
  - Simulate agent success probabilities (calibrate from V1 baseline):
    ```python
    # Example probabilities - adjust based on V1 observations
    PROBS = {
        "planner_success": 0.95,
        "coder_success_with_plan": 0.70,
        "coder_success_without_plan": 0.30,
        "tester_finds_error": 0.40,
        "debugger_fixes_error": 0.50,
    }
    ```
  - Implement `reset(task) -> state`
  - Implement `step(action) -> (next_state, reward, done)`
  - Should run ~1000 episodes per second
  - **Success**: Fast simulation that roughly matches real behavior

- [ ] **V2.5: Q-Learning Implementation**
  - Create `rl/q_learning.py`
  - Implement `QLearningAgent` class:
    - `__init__(actions, alpha=0.1, gamma=0.95, epsilon=0.1)`
    - `q_table`: defaultdict mapping (state_key, action) -> float
    - `get_q_values(state) -> dict`
    - `choose_action(state) -> action` (epsilon-greedy)
    - `update(state, action, reward, next_state, done)` (Q-learning formula)
    - `get_policy() -> dict` (best action per state)
  - Q-Learning update formula:
    ```
    Q(s,a) = Q(s,a) + α * (reward + γ * max(Q(s',:)) - Q(s,a))
    ```
  - **Success**: Q-learning agent learns in simulation

- [ ] **V2.6: Thompson Sampling Implementation**
  - Create `rl/thompson_sampling.py`
  - Implement `ThompsonSamplingAgent` class:
    - `__init__(actions)`
    - `successes`: defaultdict for alpha (Beta distribution)
    - `failures`: defaultdict for beta (Beta distribution)
    - `choose_action(state) -> action` (sample from Beta distributions)
    - `update(state, action, reward)` (update Beta parameters)
    - `get_uncertainty(state, action) -> float`
  - **Success**: Thompson Sampling explores based on uncertainty

- [ ] **V2.7: Combined Agent (Q-Learning + Thompson Sampling)**
  - Create `rl/combined_agent.py`
  - Use Q-values for exploitation, Thompson Sampling for exploration
  - Approach: Sample from uncertainty around Q-values
    ```python
    sampled_q = mean_q + (beta_sample - 0.5) * uncertainty_scale
    ```
  - Implement same interface as QLearningAgent
  - **Success**: Combined agent balances explore/exploit

- [ ] **V2.8: Training Loop**
  - Create `training/train_simulated.py`
  - Training parameters:
    ```python
    NUM_EPISODES = 5000
    EVAL_EVERY = 100
    ```
  - Track metrics per episode: total_reward, steps, success
  - Save checkpoints every EVAL_EVERY episodes
  - Generate learning curves
  - Save final Q-table to `experiments/results/q_table.json`
  - **Success**: Training completes, learning curves show improvement

- [ ] **V2.9: Learning Curve Visualization**
  - Create `visualization/learning_curves.py`
  - Plot: Episode reward over time (with smoothing)
  - Plot: Success rate over time
  - Plot: Episode length over time
  - Save to `experiments/results/learning_curves.png`
  - **Success**: Clear visualization of learning progress

### Deliverables
- `environment/state.py`
- `environment/rewards.py`
- `environment/simulated_env.py`
- `rl/q_learning.py`
- `rl/thompson_sampling.py`
- `rl/combined_agent.py`
- `training/train_simulated.py`
- `visualization/learning_curves.py`
- `experiments/results/q_table.json`
- `experiments/results/learning_curves.png`

### Rubric Alignment
- **Controller Design (10 pts)**: Q-Learning + Thompson Sampling orchestrator ✓
- **Learning Performance (15 pts)**: Learning curves, convergence ✓
- **Reward function engineering**: Documented reward structure ✓
- **State/action space design**: Documented state representation ✓

### Exit Criteria
- [ ] Simulated environment runs fast (>100 episodes/sec)
- [ ] Q-learning converges in simulation
- [ ] Thompson Sampling shows exploration behavior
- [ ] Combined agent outperforms or matches epsilon-greedy
- [ ] Learning curves generated

---

## V3: Integration & Validation (2 hours)

### Goal
Connect learned policy to real LLM pipeline and validate performance.

### Tasks

- [ ] **V3.1: Real Environment Wrapper**
  - Create `environment/coding_env.py`
  - Same interface as simulated_env but uses real agents
  - Connect to V1 agents and tools
  - Implement `reset(task)` and `step(action)`
  - **Success**: Can run episodes with real LLM

- [ ] **V3.2: Load Trained Policy**
  - Create `training/validate_real.py`
  - Load Q-table from `experiments/results/q_table.json`
  - Create agent in eval mode (no exploration)
  - **Success**: Can load and use trained policy

- [ ] **V3.3: Run Validation Episodes**
  - Run learned policy on 20-30 real episodes
  - Use same task set as baseline
  - Record: success rate, iterations, agent call distribution
  - Save to `experiments/results/validation.json`
  - **Success**: Validation data collected

- [ ] **V3.4: Comparison Analysis**
  - Create `training/evaluate.py`
  - Compare fixed policy (V1) vs learned policy (V3):
    - Success rate
    - Average iterations to success
    - Agent call distribution
    - Time per task
  - Generate comparison table
  - Save to `experiments/results/comparison.json`
  - **Success**: Clear comparison data

- [ ] **V3.5: Q-Table Analysis**
  - Create `visualization/q_table_viz.py`
  - Visualize learned Q-values as heatmap
  - Identify learned policy: "In state X, prefer action Y"
  - Interpret: What strategies did the RL learn?
  - Save to `experiments/results/q_table_heatmap.png`
  - **Success**: Can interpret what RL learned

### Deliverables
- `environment/coding_env.py`
- `training/validate_real.py`
- `training/evaluate.py`
- `visualization/q_table_viz.py`
- `experiments/results/validation.json`
- `experiments/results/comparison.json`
- `experiments/results/q_table_heatmap.png`

### Rubric Alignment
- **Learning Performance (15 pts)**: Real validation results ✓
- **Analysis Depth (15 pts)**: Comparison, Q-table interpretation ✓
- **Before/after comparison**: Fixed vs learned policy ✓

### Exit Criteria
- [ ] Learned policy runs with real LLM
- [ ] Validation results collected
- [ ] Comparison with baseline complete
- [ ] Q-table interpreted and visualized

---

## V4: Report & Video (4 hours)

### Goal
Complete all documentation and presentation materials.

### Tasks

- [ ] **V4.1: README.md**
  - Project overview (1 paragraph)
  - Installation instructions (step-by-step)
  - Configuration (API keys, parameters)
  - Quick start guide
  - Project structure overview
  - **Success**: Someone can clone and run the project

- [ ] **V4.2: Architecture Diagram**
  - Create diagram showing:
    - RL Orchestrator at top
    - Blackboard in middle
    - 4 agents at bottom
    - Tools connected to Executor agent
    - Arrows showing communication flow
  - Save as `docs/architecture.png`
  - **Success**: Clear visual of system design

- [ ] **V4.3: Technical Report - Section 1: Introduction**
  - Problem statement: Why optimize agent orchestration?
  - Approach overview: Q-Learning + Thompson Sampling
  - Contributions summary
  - ~0.5 pages

- [ ] **V4.4: Technical Report - Section 2: System Architecture**
  - Include architecture diagram
  - Describe each agent's role
  - Describe communication protocol (Blackboard)
  - Describe tools (especially custom Complexity Analyzer)
  - ~1 page

- [ ] **V4.5: Technical Report - Section 3: RL Formulation**
  - State space definition with justification
  - Action space definition
  - Reward function with justification for each component
  - Q-Learning formula and explanation
  - Thompson Sampling formula and explanation
  - How they work together
  - ~1.5 pages

- [ ] **V4.6: Technical Report - Section 4: Experimental Setup**
  - Training methodology (simulated + real)
  - Task dataset description
  - Hyperparameters (alpha, gamma, episodes)
  - Evaluation metrics
  - ~0.5 pages

- [ ] **V4.7: Technical Report - Section 5: Results & Analysis**
  - Learning curves figure + analysis
  - Q-table heatmap + interpretation
  - Comparison table: fixed vs learned
  - What strategies did RL learn?
  - Failure case examples
  - ~1.5 pages

- [ ] **V4.8: Technical Report - Section 6: Discussion**
  - Challenges encountered and solutions
  - Limitations of current approach
  - ~0.5 pages

- [ ] **V4.9: Technical Report - Section 7: Ethical Considerations**
  - Automated code generation risks
  - Potential for misuse
  - Bias in training tasks
  - ~0.25 pages

- [ ] **V4.10: Technical Report - Section 8: Future Work**
  - Deep Q-Networks for richer state
  - More agents (Refactorer, Documenter)
  - Transfer learning between task types
  - ~0.25 pages

- [ ] **V4.11: Compile Report PDF**
  - Compile all sections into single PDF
  - Add references if needed
  - Format nicely with figures
  - Save as `docs/report.pdf`
  - Target length: ~6-8 pages
  - **Success**: Complete, professional report

- [ ] **V4.12: Demo Script**
  - Create `demo/demo.py`
  - Interactive demo that shows:
    1. A task being solved with fixed policy
    2. Same task being solved with learned policy
    3. Side-by-side comparison
  - Clear console output
  - **Success**: Compelling demo of learning

- [ ] **V4.13: Record Video**
  - 10-minute video covering:
    - (0-1 min) Introduction and motivation
    - (1-3 min) System architecture walkthrough
    - (3-5 min) RL approach explanation
    - (5-8 min) Live demo with before/after comparison
    - (8-10 min) Results and conclusions
  - Can use OBS or simple screen recording
  - Voiceover explaining what's happening
  - Save as `demo/video.mp4` or upload to YouTube
  - **Success**: Clear, professional video

- [ ] **V4.14: GitHub Repository Cleanup**
  - Remove any sensitive data (API keys)
  - Add `.gitignore`
  - Ensure all files are committed
  - Add LICENSE file (MIT)
  - Verify README is complete
  - **Success**: Repository is public-ready

### Deliverables
- `README.md`
- `docs/architecture.png`
- `docs/report.pdf`
- `demo/demo.py`
- `demo/video.mp4` (or YouTube link)
- Clean GitHub repository

### Rubric Alignment
- **Technical Documentation (5 pts)**: README, report, architecture diagram ✓
- **Presentation Quality (5 pts)**: Video, visualizations ✓
- **Clarity and completeness**: Full report with all sections ✓
- **Reproducibility**: README with setup instructions ✓

### Exit Criteria
- [ ] README complete with setup instructions
- [ ] Architecture diagram created
- [ ] Report PDF complete (~6-8 pages)
- [ ] Video recorded (10 minutes)
- [ ] GitHub repo cleaned and organized

---

## V5: Buffer (2 hours)

Reserved for:
- Bug fixes discovered during video recording
- Report revisions
- Any unexpected issues
- Final polish and review

---

## Coding Tasks Dataset

Use these tasks for training and evaluation:

### Simple (Baseline)
1. Write a function that returns the sum of two numbers
2. Write a function that reverses a string
3. Write a function that checks if a number is even
4. Write a function that finds the maximum in a list
5. Write a function that counts vowels in a string

### Medium
6. Write a function that checks if a string is a palindrome
7. Write a function that computes factorial recursively
8. Write a function that returns Fibonacci sequence up to n
9. Write a function that merges two sorted lists
10. Write a function that removes duplicates from a list

### Harder (Stretch)
11. Write a function that validates balanced parentheses
12. Write a function that implements binary search
13. Write a function that sorts a list using bubble sort
14. Write a function that finds the longest common prefix
15. Write a function that rotates a list by k positions

---

## Configuration Reference

### config.yaml
```yaml
# API Configuration
api:
  provider: "openrouter"
  base_url: "https://openrouter.ai/api/v1"
  model: "x-ai/grok-beta"  # or other free model
  api_key: "${OPENROUTER_API_KEY}"  # Use environment variable
  max_tokens: 1024
  temperature: 0.7

# RL Hyperparameters
rl:
  alpha: 0.1          # Learning rate
  gamma: 0.95         # Discount factor
  epsilon: 0.1        # Exploration rate (for epsilon-greedy baseline)
  num_episodes: 5000  # Training episodes
  eval_every: 100     # Evaluation frequency

# Environment
environment:
  max_iterations: 5   # Max steps per episode
  code_timeout: 5     # Seconds to run code

# Simulation probabilities (calibrate from real data)
simulation:
  planner_success: 0.95
  coder_success_with_plan: 0.70
  coder_success_without_plan: 0.30
  tester_finds_error: 0.40
  debugger_fixes_error: 0.50
```

---

## Key Formulas (for Report)

### Q-Learning Update
```
Q(s, a) ← Q(s, a) + α [r + γ max_a' Q(s', a') - Q(s, a)]
```

Where:
- `α` = learning rate
- `γ` = discount factor
- `r` = immediate reward
- `s'` = next state
- `max_a' Q(s', a')` = maximum Q-value over all actions in next state

### Thompson Sampling
For each state-action pair, maintain Beta distribution:
```
θ(s, a) ~ Beta(α_sa, β_sa)
```

Action selection:
```
a* = argmax_a  θ_sample(s, a)
```

Update after reward:
```
If reward > 0: α_sa ← α_sa + reward
If reward ≤ 0: β_sa ← β_sa + |reward|
```

---

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| API rate limiting | Add delays between calls, batch training in simulation |
| Code execution hangs | Use subprocess with timeout |
| LLM outputs explanation instead of code | Improve prompt: "Output ONLY Python code, no explanations" |
| Q-learning not converging | Check reward function, increase episodes, verify state representation |
| Thompson Sampling over-explores | Increase prior strength (start with higher α, β) |

---

## Success Metrics

### Minimum Viable Submission
- [ ] 2 RL methods implemented and working
- [ ] 4 agents with communication
- [ ] 1 custom tool (Complexity Analyzer)
- [ ] Learning curves showing improvement
- [ ] Before/after comparison
- [ ] Report PDF
- [ ] 10-minute video
- [ ] GitHub repository

### Target Performance
- Learned policy success rate ≥ fixed policy success rate
- Learned policy uses fewer average iterations
- Clear learning curve convergence
- Interpretable Q-table showing sensible policy

---

## Final Checklist Before Submission

- [ ] All code runs without errors
- [ ] README has complete setup instructions
- [ ] No API keys in committed code
- [ ] Report PDF is complete and formatted
- [ ] Video is ~10 minutes and covers all required points
- [ ] GitHub repo is public and organized
- [ ] All experimental results are saved
- [ ] Visualizations are included in report

---

## Notes for Claude Code

When implementing this project:

1. **Start with V0** - Always verify API works before building anything complex
2. **Test incrementally** - Each component should work before moving on
3. **Keep it simple** - Tabular Q-learning is sufficient, don't over-engineer
4. **Simulation first** - Train on simulation, validate on real LLM
5. **Document as you go** - Don't leave all documentation for the end
6. **Save intermediate results** - Always save to files, don't rely on memory
7. **Use meaningful commits** - Track progress with clear commit messages

Good luck! 🚀
