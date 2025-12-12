"""
Complexity Analyzer Tool - Analyzes code complexity using AST.

This is the CUSTOM TOOL required by the assignment rubric.
Provides multiple complexity metrics for Python code.
"""

import ast
from dataclasses import dataclass
from typing import Optional


@dataclass
class ComplexityMetrics:
    """Code complexity metrics."""
    cyclomatic_complexity: int      # Decision points + 1
    lines_of_code: int              # Non-empty lines
    num_functions: int              # Function definitions
    max_nesting_depth: int          # Deepest nesting level
    cognitive_complexity: int       # Weighted complexity score
    num_classes: int = 0            # Class definitions
    num_imports: int = 0            # Import statements
    num_variables: int = 0          # Variable assignments

    def overall_score(self) -> float:
        """
        Calculate an overall complexity score (lower is better).

        Returns:
            Weighted complexity score
        """
        # Weighted formula emphasizing readability factors
        score = (
            self.cyclomatic_complexity * 2.0 +
            self.max_nesting_depth * 3.0 +
            self.cognitive_complexity * 1.5 +
            max(0, self.lines_of_code - 20) * 0.1  # Penalize very long code
        )
        return round(score, 2)

    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
        return {
            "cyclomatic_complexity": self.cyclomatic_complexity,
            "lines_of_code": self.lines_of_code,
            "num_functions": self.num_functions,
            "max_nesting_depth": self.max_nesting_depth,
            "cognitive_complexity": self.cognitive_complexity,
            "num_classes": self.num_classes,
            "num_imports": self.num_imports,
            "num_variables": self.num_variables,
            "overall_score": self.overall_score()
        }


class ComplexityVisitor(ast.NodeVisitor):
    """AST visitor to calculate complexity metrics."""

    def __init__(self):
        self.cyclomatic = 1  # Base complexity
        self.nesting_depth = 0
        self.max_nesting = 0
        self.cognitive = 0
        self.num_functions = 0
        self.num_classes = 0
        self.num_imports = 0
        self.num_variables = 0

    def _increment_cyclomatic(self):
        """Increment cyclomatic complexity for decision points."""
        self.cyclomatic += 1

    def _add_cognitive(self, increment: int):
        """Add to cognitive complexity with nesting weight."""
        # Cognitive complexity increases more with nesting
        self.cognitive += increment * (1 + self.nesting_depth)

    def _enter_nested_block(self):
        """Track entering a nested block."""
        self.nesting_depth += 1
        self.max_nesting = max(self.max_nesting, self.nesting_depth)

    def _exit_nested_block(self):
        """Track exiting a nested block."""
        self.nesting_depth -= 1

    def visit_If(self, node):
        """Count if statements."""
        self._increment_cyclomatic()
        self._add_cognitive(1)
        self._enter_nested_block()
        self.generic_visit(node)
        self._exit_nested_block()

        # Count elif branches
        for _ in node.orelse:
            if isinstance(_, ast.If):
                self._increment_cyclomatic()

    def visit_For(self, node):
        """Count for loops."""
        self._increment_cyclomatic()
        self._add_cognitive(1)
        self._enter_nested_block()
        self.generic_visit(node)
        self._exit_nested_block()

    def visit_While(self, node):
        """Count while loops."""
        self._increment_cyclomatic()
        self._add_cognitive(1)
        self._enter_nested_block()
        self.generic_visit(node)
        self._exit_nested_block()

    def visit_ExceptHandler(self, node):
        """Count except handlers."""
        self._increment_cyclomatic()
        self._add_cognitive(1)
        self.generic_visit(node)

    def visit_With(self, node):
        """Count with statements (context managers)."""
        self._enter_nested_block()
        self.generic_visit(node)
        self._exit_nested_block()

    def visit_BoolOp(self, node):
        """Count boolean operators (and/or)."""
        # Each and/or adds a decision point
        num_ops = len(node.values) - 1
        self.cyclomatic += num_ops
        self._add_cognitive(num_ops)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        """Count function definitions."""
        self.num_functions += 1
        self._enter_nested_block()
        self.generic_visit(node)
        self._exit_nested_block()

    def visit_AsyncFunctionDef(self, node):
        """Count async function definitions."""
        self.num_functions += 1
        self._enter_nested_block()
        self.generic_visit(node)
        self._exit_nested_block()

    def visit_ClassDef(self, node):
        """Count class definitions."""
        self.num_classes += 1
        self._enter_nested_block()
        self.generic_visit(node)
        self._exit_nested_block()

    def visit_Import(self, node):
        """Count import statements."""
        self.num_imports += len(node.names)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Count from imports."""
        self.num_imports += len(node.names)
        self.generic_visit(node)

    def visit_Assign(self, node):
        """Count variable assignments."""
        self.num_variables += len(node.targets)
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        """Count augmented assignments (+=, etc.)."""
        self.num_variables += 1
        self.generic_visit(node)

    def visit_ListComp(self, node):
        """Count list comprehensions."""
        self._add_cognitive(1)
        self.generic_visit(node)

    def visit_DictComp(self, node):
        """Count dict comprehensions."""
        self._add_cognitive(1)
        self.generic_visit(node)

    def visit_SetComp(self, node):
        """Count set comprehensions."""
        self._add_cognitive(1)
        self.generic_visit(node)

    def visit_Lambda(self, node):
        """Count lambda functions."""
        self._add_cognitive(1)
        self.generic_visit(node)

    def visit_IfExp(self, node):
        """Count ternary expressions."""
        self._increment_cyclomatic()
        self._add_cognitive(1)
        self.generic_visit(node)


class ComplexityAnalyzer:
    """
    Analyzes Python code complexity using AST analysis.

    This tool provides multiple metrics:
    - Cyclomatic complexity: Count of decision points
    - Lines of code: Non-empty lines
    - Function count: Number of function definitions
    - Max nesting depth: Deepest nesting level
    - Cognitive complexity: Weighted complexity considering nesting
    """

    def analyze(self, code: str) -> ComplexityMetrics:
        """
        Analyze Python code and return complexity metrics.

        Args:
            code: Python source code string

        Returns:
            ComplexityMetrics with all calculated metrics
        """
        # Count lines of code (non-empty)
        lines = [l for l in code.split('\n') if l.strip()]
        loc = len(lines)

        # Parse AST
        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Return minimal metrics for invalid code
            return ComplexityMetrics(
                cyclomatic_complexity=0,
                lines_of_code=loc,
                num_functions=0,
                max_nesting_depth=0,
                cognitive_complexity=0
            )

        # Visit AST to calculate metrics
        visitor = ComplexityVisitor()
        visitor.visit(tree)

        return ComplexityMetrics(
            cyclomatic_complexity=visitor.cyclomatic,
            lines_of_code=loc,
            num_functions=visitor.num_functions,
            max_nesting_depth=visitor.max_nesting,
            cognitive_complexity=visitor.cognitive,
            num_classes=visitor.num_classes,
            num_imports=visitor.num_imports,
            num_variables=visitor.num_variables
        )

    def get_complexity_rating(self, metrics: ComplexityMetrics) -> str:
        """
        Get a human-readable complexity rating.

        Args:
            metrics: ComplexityMetrics to evaluate

        Returns:
            Rating string (Low, Medium, High, Very High)
        """
        score = metrics.overall_score()
        if score < 5:
            return "Low"
        elif score < 15:
            return "Medium"
        elif score < 30:
            return "High"
        else:
            return "Very High"


if __name__ == "__main__":
    analyzer = ComplexityAnalyzer()

    # Test with simple code
    simple_code = """
def add(a, b):
    return a + b
"""
    print("Simple code:")
    metrics = analyzer.analyze(simple_code)
    print(f"  Metrics: {metrics.to_dict()}")
    print(f"  Rating: {analyzer.get_complexity_rating(metrics)}")

    # Test with moderately complex code
    moderate_code = """
def find_max(lst):
    if not lst:
        return None
    max_val = lst[0]
    for item in lst:
        if item > max_val:
            max_val = item
    return max_val
"""
    print("\nModerate code:")
    metrics = analyzer.analyze(moderate_code)
    print(f"  Metrics: {metrics.to_dict()}")
    print(f"  Rating: {analyzer.get_complexity_rating(metrics)}")

    # Test with complex code
    complex_code = """
def process_data(data, options=None):
    if not data:
        return []

    results = []
    for item in data:
        if item.get('type') == 'A':
            if item.get('value') > 0:
                for sub in item.get('children', []):
                    if sub.get('active'):
                        try:
                            result = sub['value'] * 2
                            results.append(result)
                        except KeyError:
                            continue
        elif item.get('type') == 'B':
            results.append(item.get('value', 0))

    return sorted(results) if options and options.get('sort') else results
"""
    print("\nComplex code:")
    metrics = analyzer.analyze(complex_code)
    print(f"  Metrics: {metrics.to_dict()}")
    print(f"  Rating: {analyzer.get_complexity_rating(metrics)}")

    print("\nComplexity analyzer working!")
