"""
Unit tests for the tools: CodeExecutor, ComplexityAnalyzer, TestRunner.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from tools.code_executor import CodeExecutor, ExecutionResult
from tools.complexity_analyzer import ComplexityAnalyzer, ComplexityMetrics, ComplexityVisitor
from tools.test_runner import TestRunner, TestResult, TestCase


class TestExecutionResult(unittest.TestCase):
    """Test ExecutionResult dataclass."""

    def test_creation(self):
        """Test creating an execution result."""
        result = ExecutionResult(
            success=True,
            output="5",
            error=""
        )
        self.assertTrue(result.success)
        self.assertEqual(result.output, "5")
        self.assertEqual(result.error, "")

    def test_with_return_value(self):
        """Test execution result with return value."""
        result = ExecutionResult(
            success=True,
            output="",
            error="",
            return_value="42"
        )
        self.assertEqual(result.return_value, "42")


class TestCodeExecutor(unittest.TestCase):
    """Test CodeExecutor class."""

    def setUp(self):
        """Create executor for each test."""
        self.executor = CodeExecutor(timeout=5)

    def test_execute_simple_code(self):
        """Test executing simple Python code."""
        code = "print('hello world')"
        result = self.executor.execute(code)
        self.assertTrue(result.success)
        self.assertEqual(result.output, "hello world")
        self.assertEqual(result.error, "")

    def test_execute_with_computation(self):
        """Test executing code with computation."""
        code = """
x = 2 + 3
print(x)
"""
        result = self.executor.execute(code)
        self.assertTrue(result.success)
        self.assertEqual(result.output, "5")

    def test_execute_function(self):
        """Test executing a function definition and call."""
        code = """
def add(a, b):
    return a + b

print(add(2, 3))
"""
        result = self.executor.execute(code)
        self.assertTrue(result.success)
        self.assertEqual(result.output, "5")

    def test_execute_with_test_code(self):
        """Test executing code with appended test code."""
        code = """
def multiply(a, b):
    return a * b
"""
        test_code = "print(multiply(3, 4))"
        result = self.executor.execute(code, test_code)
        self.assertTrue(result.success)
        self.assertEqual(result.output, "12")

    def test_execute_syntax_error(self):
        """Test executing code with syntax error."""
        code = "def foo( print('test')"
        result = self.executor.execute(code)
        self.assertFalse(result.success)
        self.assertIn("SyntaxError", result.error)

    def test_execute_runtime_error(self):
        """Test executing code with runtime error."""
        code = "print(undefined_variable)"
        result = self.executor.execute(code)
        self.assertFalse(result.success)
        self.assertIn("NameError", result.error)

    def test_execute_timeout(self):
        """Test that infinite loop times out."""
        executor = CodeExecutor(timeout=1)
        code = "while True: pass"
        result = executor.execute(code)
        self.assertFalse(result.success)
        self.assertIn("timed out", result.error.lower())

    def test_check_syntax_valid(self):
        """Test syntax check on valid code."""
        code = "def foo(): return 42"
        valid, error = self.executor.check_syntax(code)
        self.assertTrue(valid)
        self.assertEqual(error, "")

    def test_check_syntax_invalid(self):
        """Test syntax check on invalid code."""
        code = "def foo( return 42"
        valid, error = self.executor.check_syntax(code)
        self.assertFalse(valid)
        self.assertIn("Syntax error", error)

    def test_run_with_inputs(self):
        """Test running function with multiple inputs."""
        code = """
def add(a, b):
    return a + b
"""
        results = self.executor.run_with_inputs(
            code, "add", [(1, 2), (5, 5), (0, 0)]
        )
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertTrue(result.success)

    def test_run_with_single_input(self):
        """Test running function with single argument."""
        code = """
def square(x):
    return x * x
"""
        results = self.executor.run_with_inputs(code, "square", [2, 3, 4])
        self.assertEqual(len(results), 3)
        self.assertIn("Result: 4", results[0].output)
        self.assertIn("Result: 9", results[1].output)
        self.assertIn("Result: 16", results[2].output)


class TestComplexityMetrics(unittest.TestCase):
    """Test ComplexityMetrics dataclass."""

    def test_creation(self):
        """Test creating complexity metrics."""
        metrics = ComplexityMetrics(
            cyclomatic_complexity=5,
            lines_of_code=20,
            num_functions=2,
            max_nesting_depth=3,
            cognitive_complexity=10
        )
        self.assertEqual(metrics.cyclomatic_complexity, 5)
        self.assertEqual(metrics.lines_of_code, 20)
        self.assertEqual(metrics.num_functions, 2)

    def test_overall_score(self):
        """Test overall score calculation."""
        metrics = ComplexityMetrics(
            cyclomatic_complexity=5,
            lines_of_code=10,
            num_functions=1,
            max_nesting_depth=2,
            cognitive_complexity=4
        )
        score = metrics.overall_score()
        # score = 5*2 + 2*3 + 4*1.5 + max(0, 10-20)*0.1
        # score = 10 + 6 + 6 + 0 = 22
        self.assertEqual(score, 22.0)

    def test_overall_score_long_code(self):
        """Test overall score penalizes long code."""
        metrics = ComplexityMetrics(
            cyclomatic_complexity=1,
            lines_of_code=50,
            num_functions=1,
            max_nesting_depth=1,
            cognitive_complexity=0
        )
        score = metrics.overall_score()
        # Includes penalty for lines > 20
        # score = 1*2 + 1*3 + 0*1.5 + max(0, 50-20)*0.1 = 2 + 3 + 0 + 3 = 8
        self.assertEqual(score, 8.0)

    def test_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = ComplexityMetrics(
            cyclomatic_complexity=3,
            lines_of_code=15,
            num_functions=1,
            max_nesting_depth=2,
            cognitive_complexity=4,
            num_classes=1,
            num_imports=2,
            num_variables=5
        )
        d = metrics.to_dict()
        self.assertEqual(d["cyclomatic_complexity"], 3)
        self.assertEqual(d["num_classes"], 1)
        self.assertEqual(d["num_imports"], 2)
        self.assertIn("overall_score", d)


class TestComplexityAnalyzer(unittest.TestCase):
    """Test ComplexityAnalyzer class."""

    def setUp(self):
        """Create analyzer for each test."""
        self.analyzer = ComplexityAnalyzer()

    def test_analyze_simple_function(self):
        """Test analyzing a simple function."""
        code = """
def add(a, b):
    return a + b
"""
        metrics = self.analyzer.analyze(code)
        self.assertEqual(metrics.num_functions, 1)
        self.assertEqual(metrics.cyclomatic_complexity, 1)
        self.assertEqual(metrics.max_nesting_depth, 1)

    def test_analyze_with_if_statements(self):
        """Test analyzing code with if statements."""
        code = """
def check(x):
    if x > 0:
        return "positive"
    elif x < 0:
        return "negative"
    else:
        return "zero"
"""
        metrics = self.analyzer.analyze(code)
        # if + elif = 2 decision points
        self.assertGreaterEqual(metrics.cyclomatic_complexity, 2)
        self.assertEqual(metrics.num_functions, 1)

    def test_analyze_with_loops(self):
        """Test analyzing code with loops."""
        code = """
def sum_list(lst):
    total = 0
    for item in lst:
        total += item
    return total
"""
        metrics = self.analyzer.analyze(code)
        # for loop = 1 decision point
        self.assertGreaterEqual(metrics.cyclomatic_complexity, 2)

    def test_analyze_nested_code(self):
        """Test analyzing deeply nested code."""
        code = """
def nested(x):
    if x > 0:
        for i in range(x):
            if i % 2 == 0:
                print(i)
"""
        metrics = self.analyzer.analyze(code)
        # Should detect nesting depth > 1
        self.assertGreater(metrics.max_nesting_depth, 1)

    def test_analyze_with_try_except(self):
        """Test analyzing code with try/except."""
        code = """
def safe_divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return None
"""
        metrics = self.analyzer.analyze(code)
        # except adds to cyclomatic complexity
        self.assertGreaterEqual(metrics.cyclomatic_complexity, 2)

    def test_analyze_with_class(self):
        """Test analyzing code with a class."""
        code = """
class Calculator:
    def add(self, a, b):
        return a + b

    def multiply(self, a, b):
        return a * b
"""
        metrics = self.analyzer.analyze(code)
        self.assertEqual(metrics.num_classes, 1)
        self.assertEqual(metrics.num_functions, 2)

    def test_analyze_with_imports(self):
        """Test analyzing code with imports."""
        code = """
import os
from sys import path, argv
import json

def foo():
    pass
"""
        metrics = self.analyzer.analyze(code)
        # os (1) + path,argv (2) + json (1) = 4
        self.assertEqual(metrics.num_imports, 4)

    def test_analyze_syntax_error(self):
        """Test analyzing code with syntax error."""
        code = "def foo( print('test')"
        metrics = self.analyzer.analyze(code)
        # Should return minimal metrics for invalid code
        self.assertEqual(metrics.cyclomatic_complexity, 0)

    def test_analyze_lines_of_code(self):
        """Test lines of code count."""
        code = """
def foo():
    x = 1

    y = 2

    return x + y
"""
        metrics = self.analyzer.analyze(code)
        # Non-empty lines: def, x=1, y=2, return = 4
        self.assertEqual(metrics.lines_of_code, 4)

    def test_analyze_boolean_operators(self):
        """Test analyzing code with boolean operators."""
        code = """
def check(a, b, c):
    if a and b or c:
        return True
    return False
"""
        metrics = self.analyzer.analyze(code)
        # if (1) + and (1) + or (1) = adds to complexity
        self.assertGreaterEqual(metrics.cyclomatic_complexity, 3)

    def test_get_complexity_rating_low(self):
        """Test low complexity rating (score < 5)."""
        # Score formula: cyclo*2 + nesting*3 + cognitive*1.5 + max(0, loc-20)*0.1
        # Target: score < 5
        # Using: 1*2 + 0*3 + 0*1.5 + 0 = 2
        metrics = ComplexityMetrics(
            cyclomatic_complexity=1,
            lines_of_code=5,
            num_functions=1,
            max_nesting_depth=0,
            cognitive_complexity=0
        )
        rating = self.analyzer.get_complexity_rating(metrics)
        self.assertEqual(rating, "Low")

    def test_get_complexity_rating_medium(self):
        """Test medium complexity rating (5 <= score < 15)."""
        # Score formula: cyclo*2 + nesting*3 + cognitive*1.5 + max(0, loc-20)*0.1
        # Target: 5 <= score < 15
        # Using: 2*2 + 1*3 + 2*1.5 + 0 = 4 + 3 + 3 = 10
        metrics = ComplexityMetrics(
            cyclomatic_complexity=2,
            lines_of_code=10,
            num_functions=1,
            max_nesting_depth=1,
            cognitive_complexity=2
        )
        rating = self.analyzer.get_complexity_rating(metrics)
        self.assertEqual(rating, "Medium")

    def test_get_complexity_rating_high(self):
        """Test high complexity rating (15 <= score < 30)."""
        # Score formula: cyclo*2 + nesting*3 + cognitive*1.5 + max(0, loc-20)*0.1
        # Target: 15 <= score < 30
        # Using: 5*2 + 2*3 + 4*1.5 + 0 = 10 + 6 + 6 = 22
        metrics = ComplexityMetrics(
            cyclomatic_complexity=5,
            lines_of_code=15,
            num_functions=2,
            max_nesting_depth=2,
            cognitive_complexity=4
        )
        rating = self.analyzer.get_complexity_rating(metrics)
        self.assertEqual(rating, "High")


class TestTestCase(unittest.TestCase):
    """Test TestCase dataclass."""

    def test_creation(self):
        """Test creating a test case."""
        tc = TestCase(
            inputs=(1, 2),
            expected=3,
            description="add two numbers"
        )
        self.assertEqual(tc.inputs, (1, 2))
        self.assertEqual(tc.expected, 3)
        self.assertEqual(tc.description, "add two numbers")

    def test_default_description(self):
        """Test default empty description."""
        tc = TestCase(inputs=(1,), expected=1)
        self.assertEqual(tc.description, "")


class TestTestResult(unittest.TestCase):
    """Test TestResult dataclass."""

    def test_creation(self):
        """Test creating a test result."""
        result = TestResult(
            tests_passed=3,
            tests_failed=1,
            total_tests=4
        )
        self.assertEqual(result.tests_passed, 3)
        self.assertEqual(result.tests_failed, 1)
        self.assertEqual(result.total_tests, 4)

    def test_success_rate(self):
        """Test success rate calculation."""
        result = TestResult(tests_passed=3, tests_failed=1, total_tests=4)
        self.assertEqual(result.success_rate, 0.75)

    def test_success_rate_zero_tests(self):
        """Test success rate with no tests."""
        result = TestResult(tests_passed=0, tests_failed=0, total_tests=0)
        self.assertEqual(result.success_rate, 0.0)

    def test_all_passed_true(self):
        """Test all_passed when all tests pass."""
        result = TestResult(tests_passed=4, tests_failed=0, total_tests=4)
        self.assertTrue(result.all_passed)

    def test_all_passed_false_with_failures(self):
        """Test all_passed when some tests fail."""
        result = TestResult(tests_passed=3, tests_failed=1, total_tests=4)
        self.assertFalse(result.all_passed)

    def test_all_passed_false_with_no_passes(self):
        """Test all_passed when no tests pass."""
        result = TestResult(tests_passed=0, tests_failed=0, total_tests=0)
        self.assertFalse(result.all_passed)


class TestTestRunner(unittest.TestCase):
    """Test TestRunner class."""

    def setUp(self):
        """Create test runner for each test."""
        self.runner = TestRunner(timeout=5)

    def test_detect_task_type_sum(self):
        """Test detecting sum task type."""
        task = "Write a function that returns the sum of two numbers"
        self.assertEqual(self.runner.detect_task_type(task), "sum")

    def test_detect_task_type_reverse(self):
        """Test detecting reverse task type."""
        task = "Write a function that reverses a string"
        self.assertEqual(self.runner.detect_task_type(task), "reverse")

    def test_detect_task_type_even(self):
        """Test detecting even task type."""
        task = "Write a function that checks if a number is even"
        self.assertEqual(self.runner.detect_task_type(task), "even")

    def test_detect_task_type_max(self):
        """Test detecting max task type."""
        task = "Write a function that finds the maximum in a list"
        self.assertEqual(self.runner.detect_task_type(task), "max")

    def test_detect_task_type_unknown(self):
        """Test detecting unknown task type."""
        task = "Write a function that does something weird"
        self.assertIsNone(self.runner.detect_task_type(task))

    def test_extract_function_name(self):
        """Test extracting function name from code."""
        code = """
def add(a, b):
    return a + b
"""
        name = self.runner.extract_function_name(code)
        self.assertEqual(name, "add")

    def test_extract_function_name_complex(self):
        """Test extracting function name from more complex code."""
        code = """
# A comment
import math

def calculate_area(radius):
    return math.pi * radius ** 2
"""
        name = self.runner.extract_function_name(code)
        self.assertEqual(name, "calculate_area")

    def test_extract_function_name_no_function(self):
        """Test extracting function name when no function exists."""
        code = "x = 1 + 2"
        name = self.runner.extract_function_name(code)
        self.assertIsNone(name)

    def test_run_tests_sum_correct(self):
        """Test running tests on correct sum function."""
        code = """
def add(a, b):
    return a + b
"""
        result = self.runner.run_tests(
            code, "Write a function that returns the sum of two numbers"
        )
        self.assertTrue(result.all_passed)
        self.assertEqual(result.tests_passed, result.total_tests)

    def test_run_tests_sum_incorrect(self):
        """Test running tests on incorrect sum function."""
        code = """
def add(a, b):
    return a - b  # Bug: subtracts instead of adds
"""
        result = self.runner.run_tests(
            code, "Write a function that returns the sum of two numbers"
        )
        self.assertFalse(result.all_passed)
        self.assertGreater(result.tests_failed, 0)

    def test_run_tests_syntax_error(self):
        """Test running tests on code with syntax error."""
        code = "def add(a, b) return a + b"
        result = self.runner.run_tests(code, "sum task")
        self.assertFalse(result.all_passed)
        self.assertEqual(result.tests_passed, 0)
        self.assertIn("Syntax error", result.error_messages[0])

    def test_run_tests_no_function(self):
        """Test running tests when no function is defined."""
        code = "x = 1 + 2"
        result = self.runner.run_tests(code, "sum task")
        self.assertFalse(result.all_passed)
        self.assertIn("Could not find function definition", result.error_messages[0])

    def test_run_tests_custom_tests(self):
        """Test running custom test cases."""
        code = """
def double(x):
    return x * 2
"""
        custom_tests = [
            TestCase((2,), 4, "double 2"),
            TestCase((5,), 10, "double 5"),
            TestCase((0,), 0, "double 0"),
        ]
        result = self.runner.run_tests(code, "unknown task", custom_tests)
        self.assertTrue(result.all_passed)
        self.assertEqual(result.tests_passed, 3)

    def test_run_tests_unknown_task_code_runs(self):
        """Test running tests when task is unknown but code runs."""
        code = """
def mystery():
    return 42
"""
        result = self.runner.run_tests(code, "something unknown")
        # Should check if code runs successfully
        self.assertEqual(result.total_tests, 1)

    def test_run_tests_reverse_correct(self):
        """Test running tests on correct reverse function."""
        code = """
def reverse(s):
    return s[::-1]
"""
        result = self.runner.run_tests(
            code, "Write a function that reverses a string"
        )
        self.assertTrue(result.all_passed)

    def test_run_tests_max_correct(self):
        """Test running tests on correct max function."""
        code = """
def find_max(lst):
    return max(lst)
"""
        result = self.runner.run_tests(
            code, "Write a function that finds the maximum in a list"
        )
        self.assertTrue(result.all_passed)

    def test_run_tests_with_details(self):
        """Test that test results include details."""
        code = """
def add(a, b):
    return a + b
"""
        result = self.runner.run_tests(code, "sum task")
        self.assertIsInstance(result.details, list)
        self.assertGreater(len(result.details), 0)
        # Check detail structure
        detail = result.details[0]
        self.assertIn("test", detail)
        self.assertIn("passed", detail)


class TestToolsIntegration(unittest.TestCase):
    """Integration tests combining multiple tools."""

    def test_executor_and_analyzer_together(self):
        """Test using executor and analyzer on same code."""
        code = """
def process(lst):
    result = []
    for item in lst:
        if item > 0:
            result.append(item * 2)
    return result
"""
        executor = CodeExecutor()
        analyzer = ComplexityAnalyzer()

        # Check syntax
        valid, _ = executor.check_syntax(code)
        self.assertTrue(valid)

        # Analyze complexity
        metrics = analyzer.analyze(code)
        self.assertEqual(metrics.num_functions, 1)
        self.assertGreater(metrics.cyclomatic_complexity, 1)

    def test_runner_with_complexity_check(self):
        """Test using runner and checking complexity of solution."""
        code = """
def add(a, b):
    return a + b
"""
        runner = TestRunner()
        analyzer = ComplexityAnalyzer()

        # Run tests
        result = runner.run_tests(code, "sum task")
        self.assertTrue(result.all_passed)

        # Check complexity - simple function should have low score
        # Score = cyclo*2 + nesting*3 + cognitive*1.5 + loc penalty
        # For simple add: cyclo=1, nesting=1 (function), cognitive=0
        # Score = 1*2 + 1*3 + 0*1.5 = 5, which is the boundary for Medium
        metrics = analyzer.analyze(code)
        rating = analyzer.get_complexity_rating(metrics)
        # The simple function gets "Medium" rating due to function nesting
        self.assertIn(rating, ["Low", "Medium"])


if __name__ == "__main__":
    unittest.main()
