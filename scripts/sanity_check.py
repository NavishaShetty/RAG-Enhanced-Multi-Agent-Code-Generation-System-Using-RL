"""
V0.2: Sanity Check Script
Tests 5 simple coding tasks with the LLM to verify code generation works.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.api import OpenRouterClient


CODING_TASKS = [
    "Write a Python function that returns the sum of two numbers",
    "Write a Python function that reverses a string",
    "Write a Python function that checks if a number is even",
    "Write a Python function that finds the maximum in a list",
    "Write a Python function that counts vowels in a string"
]

SYSTEM_PROMPT = """You are a coding assistant. Write clean, working Python code.
Output ONLY the Python code with no explanations. Use ```python code blocks."""


def extract_code(response: str) -> str:
    """Extract Python code from LLM response."""
    if "```python" in response:
        parts = response.split("```python")
        if len(parts) > 1:
            code_part = parts[1].split("```")[0]
            return code_part.strip()
    elif "```" in response:
        parts = response.split("```")
        if len(parts) > 1:
            return parts[1].strip()
    return response.strip()


def is_valid_python(code: str) -> bool:
    """Check if code is syntactically valid Python."""
    try:
        compile(code, "<string>", "exec")
        return True
    except SyntaxError:
        return False


def run_sanity_check():
    """Run sanity check on 5 coding tasks."""
    print("=" * 60)
    print("V0.2: Code Generation Sanity Check")
    print("=" * 60)
    print()

    try:
        client = OpenRouterClient()
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set OPENROUTER_API_KEY environment variable")
        return

    results = []

    for i, task in enumerate(CODING_TASKS, 1):
        print(f"Task {i}: {task}")
        print("-" * 50)

        try:
            response = client.call(task, system_prompt=SYSTEM_PROMPT)
            code = extract_code(response)

            print("Generated Code:")
            print(code)
            print()

            valid = is_valid_python(code)
            status = "✓ Valid Python" if valid else "✗ Invalid syntax"
            print(f"Status: {status}")

            results.append({
                "task": task,
                "code": code,
                "valid": valid
            })

        except Exception as e:
            print(f"Error: {e}")
            results.append({
                "task": task,
                "code": None,
                "valid": False,
                "error": str(e)
            })

        print()
        print("=" * 60)
        print()

    # Summary
    valid_count = sum(1 for r in results if r["valid"])
    print(f"SUMMARY: {valid_count}/{len(CODING_TASKS)} tasks produced valid Python")
    print()

    if valid_count >= 3:
        print("✓ SUCCESS: At least 3/5 produce syntactically valid Python")
        print("V0.2 Exit Criteria Met!")
    else:
        print("✗ FAILED: Less than 3/5 produced valid Python")
        print("Consider adjusting prompts or model.")

    return results


if __name__ == "__main__":
    run_sanity_check()
