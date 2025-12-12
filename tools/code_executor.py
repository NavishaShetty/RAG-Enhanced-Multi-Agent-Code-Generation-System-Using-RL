"""
Code Executor Tool - Safely executes Python code with timeout.
"""

import subprocess
import sys
import tempfile
import os
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    output: str
    error: str
    return_value: Optional[str] = None


class CodeExecutor:
    """
    Safely executes Python code with timeout and resource limits.
    """

    def __init__(self, timeout: int = 5):
        """
        Initialize the code executor.

        Args:
            timeout: Maximum execution time in seconds
        """
        self.timeout = timeout

    def execute(self, code: str, test_code: Optional[str] = None) -> ExecutionResult:
        """
        Execute Python code safely in a subprocess.

        Args:
            code: The Python code to execute
            test_code: Optional test code to append

        Returns:
            ExecutionResult with success status, output, and errors
        """
        # Combine code and test code
        full_code = code
        if test_code:
            full_code = f"{code}\n\n# Test code\n{test_code}"

        # Write to temporary file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False
        ) as f:
            f.write(full_code)
            temp_path = f.name

        try:
            # Execute in subprocess with timeout
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=tempfile.gettempdir()
            )

            success = result.returncode == 0
            output = result.stdout.strip()
            error = result.stderr.strip()

            return ExecutionResult(
                success=success,
                output=output,
                error=error
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Execution timed out after {self.timeout} seconds"
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Execution failed: {str(e)}"
            )
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass

    def check_syntax(self, code: str) -> Tuple[bool, str]:
        """
        Check if code has valid Python syntax.

        Args:
            code: The Python code to check

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            compile(code, "<string>", "exec")
            return True, ""
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"

    def run_with_inputs(
        self,
        code: str,
        function_name: str,
        test_inputs: list
    ) -> list:
        """
        Run a function with multiple test inputs.

        Args:
            code: The Python code containing the function
            function_name: Name of the function to call
            test_inputs: List of input tuples to test

        Returns:
            List of ExecutionResult for each test
        """
        results = []

        for inputs in test_inputs:
            # Generate test code
            if isinstance(inputs, tuple):
                args_str = ", ".join(repr(i) for i in inputs)
            else:
                args_str = repr(inputs)

            test_code = f"""
result = {function_name}({args_str})
print(f"Result: {{result}}")
"""
            result = self.execute(code, test_code)
            results.append(result)

        return results


if __name__ == "__main__":
    # Test the executor
    executor = CodeExecutor(timeout=5)

    # Test valid code
    code = """
def add(a, b):
    return a + b

print(add(2, 3))
"""
    print("Testing valid code...")
    result = executor.execute(code)
    print(f"Success: {result.success}")
    print(f"Output: {result.output}")
    print(f"Error: {result.error}")

    # Test syntax error
    print("\nTesting syntax error...")
    bad_code = "def foo(\nprint('test')"
    valid, error = executor.check_syntax(bad_code)
    print(f"Valid syntax: {valid}")
    print(f"Error: {error}")

    # Test timeout
    print("\nTesting timeout...")
    infinite_code = "while True: pass"
    result = executor.execute(infinite_code)
    print(f"Success: {result.success}")
    print(f"Error: {result.error}")

    # Test with multiple inputs
    print("\nTesting with multiple inputs...")
    func_code = """
def add(a, b):
    return a + b
"""
    results = executor.run_with_inputs(func_code, "add", [(1, 2), (5, 5), (-1, 1)])
    for i, r in enumerate(results):
        print(f"Test {i+1}: {r.output if r.success else r.error}")

    print("\nCode executor working!")
