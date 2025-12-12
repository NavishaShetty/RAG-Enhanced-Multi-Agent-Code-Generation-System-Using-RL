"""
Test Runner Tool - Generates and runs tests for functions.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
from dataclasses import dataclass, field
from typing import List, Tuple, Any, Optional
from tools.code_executor import CodeExecutor


@dataclass
class TestCase:
    """A single test case."""
    inputs: Tuple
    expected: Any
    description: str = ""


@dataclass
class TestResult:
    """Result of running tests."""
    tests_passed: int
    tests_failed: int
    total_tests: int
    error_messages: List[str] = field(default_factory=list)
    details: List[dict] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return self.tests_passed / self.total_tests

    @property
    def all_passed(self) -> bool:
        return self.tests_failed == 0 and self.tests_passed > 0


class TestRunner:
    """
    Generates and runs test cases for functions.
    """

    # Pre-defined test cases for common task types
    TASK_TESTS = {
        "sum": [
            TestCase((1, 2), 3, "basic addition"),
            TestCase((0, 0), 0, "zeros"),
            TestCase((-1, 1), 0, "negative and positive"),
            TestCase((100, 200), 300, "larger numbers"),
        ],
        "reverse": [
            TestCase(("hello",), "olleh", "basic string"),
            TestCase(("",), "", "empty string"),
            TestCase(("a",), "a", "single char"),
            TestCase(("12321",), "12321", "palindrome"),
        ],
        "even": [
            TestCase((2,), True, "even number"),
            TestCase((3,), False, "odd number"),
            TestCase((0,), True, "zero"),
            TestCase((-4,), True, "negative even"),
        ],
        "max": [
            TestCase(([1, 2, 3],), 3, "simple list"),
            TestCase(([5],), 5, "single element"),
            TestCase(([-1, -2, -3],), -1, "negative numbers"),
            TestCase(([3, 3, 3],), 3, "all same"),
        ],
        "vowels": [
            TestCase(("hello",), 2, "basic string"),
            TestCase(("",), 0, "empty string"),
            TestCase(("xyz",), 0, "no vowels"),
            TestCase(("aeiou",), 5, "all vowels"),
        ],
        "palindrome": [
            TestCase(("radar",), True, "palindrome"),
            TestCase(("hello",), False, "not palindrome"),
            TestCase(("",), True, "empty string"),
            TestCase(("a",), True, "single char"),
        ],
        "factorial": [
            TestCase((0,), 1, "zero"),
            TestCase((1,), 1, "one"),
            TestCase((5,), 120, "five"),
            TestCase((3,), 6, "three"),
        ],
        "fibonacci": [
            TestCase((0,), [], "zero"),
            TestCase((1,), [0], "one"),
            TestCase((5,), [0, 1, 1, 2, 3], "five"),
        ],
        "duplicates": [
            TestCase(([1, 2, 2, 3],), [1, 2, 3], "basic"),
            TestCase(([],), [], "empty"),
            TestCase(([1, 1, 1],), [1], "all same"),
        ],
        "binary_search": [
            TestCase(([1, 2, 3, 4, 5], 3), 2, "found middle"),
            TestCase(([1, 2, 3, 4, 5], 1), 0, "found first"),
            TestCase(([1, 2, 3, 4, 5], 6), -1, "not found"),
        ],
    }

    def __init__(self, timeout: int = 5):
        """
        Initialize the test runner.

        Args:
            timeout: Timeout for code execution
        """
        self.executor = CodeExecutor(timeout=timeout)

    def detect_task_type(self, task: str) -> Optional[str]:
        """
        Detect the type of task from description.

        Args:
            task: Task description

        Returns:
            Task type key or None
        """
        task_lower = task.lower()

        keywords = {
            "sum": ["sum", "add two"],
            "reverse": ["reverse", "reverses"],
            "even": ["even", "odd"],
            "max": ["max", "maximum", "largest"],
            "vowels": ["vowel", "vowels"],
            "palindrome": ["palindrome"],
            "factorial": ["factorial"],
            "fibonacci": ["fibonacci", "fib"],
            "duplicates": ["duplicate", "remove duplicate"],
            "binary_search": ["binary search"],
        }

        for task_type, words in keywords.items():
            if any(w in task_lower for w in words):
                return task_type
        return None

    def extract_function_name(self, code: str) -> Optional[str]:
        """
        Extract the main function name from code.

        Args:
            code: Python code

        Returns:
            Function name or None
        """
        # Match function definitions
        match = re.search(r'def\s+(\w+)\s*\(', code)
        if match:
            return match.group(1)
        return None

    def run_tests(
        self,
        code: str,
        task: str,
        custom_tests: Optional[List[TestCase]] = None
    ) -> TestResult:
        """
        Run tests against code.

        Args:
            code: Python code to test
            task: Task description (used to auto-generate tests)
            custom_tests: Optional custom test cases

        Returns:
            TestResult with pass/fail counts
        """
        # Check syntax first
        valid, syntax_error = self.executor.check_syntax(code)
        if not valid:
            return TestResult(
                tests_passed=0,
                tests_failed=1,
                total_tests=1,
                error_messages=[f"Syntax error: {syntax_error}"]
            )

        # Get function name
        func_name = self.extract_function_name(code)
        if not func_name:
            return TestResult(
                tests_passed=0,
                tests_failed=1,
                total_tests=1,
                error_messages=["Could not find function definition"]
            )

        # Get test cases
        if custom_tests:
            tests = custom_tests
        else:
            task_type = self.detect_task_type(task)
            tests = self.TASK_TESTS.get(task_type, [])

        if not tests:
            # No predefined tests - just check if code runs
            result = self.executor.execute(code)
            if result.success:
                return TestResult(
                    tests_passed=1,
                    tests_failed=0,
                    total_tests=1,
                    details=[{"test": "code_runs", "passed": True}]
                )
            else:
                return TestResult(
                    tests_passed=0,
                    tests_failed=1,
                    total_tests=1,
                    error_messages=[result.error]
                )

        # Run each test
        passed = 0
        failed = 0
        errors = []
        details = []

        for test in tests:
            # Build test code
            if len(test.inputs) == 1:
                args_str = repr(test.inputs[0])
            else:
                args_str = ", ".join(repr(i) for i in test.inputs)

            test_code = f"""
try:
    result = {func_name}({args_str})
    expected = {repr(test.expected)}
    if result == expected:
        print("PASS")
    else:
        print(f"FAIL: got {{result}}, expected {{expected}}")
except Exception as e:
    print(f"ERROR: {{e}}")
"""
            exec_result = self.executor.execute(code, test_code)

            if exec_result.success and "PASS" in exec_result.output:
                passed += 1
                details.append({
                    "test": test.description,
                    "inputs": test.inputs,
                    "expected": test.expected,
                    "passed": True
                })
            else:
                failed += 1
                error_msg = exec_result.output or exec_result.error
                errors.append(f"{test.description}: {error_msg}")
                details.append({
                    "test": test.description,
                    "inputs": test.inputs,
                    "expected": test.expected,
                    "passed": False,
                    "error": error_msg
                })

        return TestResult(
            tests_passed=passed,
            tests_failed=failed,
            total_tests=len(tests),
            error_messages=errors,
            details=details
        )


if __name__ == "__main__":
    runner = TestRunner()

    # Test sum function
    print("Testing sum function...")
    code = """
def add(a, b):
    return a + b
"""
    result = runner.run_tests(code, "Write a function that returns the sum of two numbers")
    print(f"Passed: {result.tests_passed}/{result.total_tests}")
    print(f"Success rate: {result.success_rate:.0%}")

    # Test with bugs
    print("\nTesting buggy max function...")
    buggy_code = """
def find_max(lst):
    return lst[0]  # Bug: doesn't actually find max
"""
    result = runner.run_tests(buggy_code, "Write a function that finds the maximum in a list")
    print(f"Passed: {result.tests_passed}/{result.total_tests}")
    if result.error_messages:
        print(f"Errors: {result.error_messages[:2]}")

    # Test correct max
    print("\nTesting correct max function...")
    correct_code = """
def find_max(lst):
    max_val = lst[0]
    for x in lst:
        if x > max_val:
            max_val = x
    return max_val
"""
    result = runner.run_tests(correct_code, "Write a function that finds the maximum in a list")
    print(f"Passed: {result.tests_passed}/{result.total_tests}")
    print(f"All passed: {result.all_passed}")

    print("\nTest runner working!")
