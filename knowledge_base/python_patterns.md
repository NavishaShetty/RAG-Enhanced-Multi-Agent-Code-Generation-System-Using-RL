# Python Design Patterns and Idioms

## Topic: List Comprehension

### Description
Concise way to create lists based on existing sequences.

### Example
```python
# Basic list comprehension
squares = [x**2 for x in range(10)]

# With condition
evens = [x for x in range(20) if x % 2 == 0]

# Nested comprehension (matrix transpose)
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
transposed = [[row[i] for row in matrix] for i in range(len(matrix[0]))]

# With multiple iterables
pairs = [(x, y) for x in range(3) for y in range(3)]
```

### Common Mistakes
- Creating overly complex comprehensions (use regular loop instead)
- Forgetting that comprehension creates new list

### Best Practice
Use comprehensions for simple transformations; use loops for complex logic.

---

## Topic: Dictionary Comprehension

### Description
Create dictionaries using comprehension syntax.

### Example
```python
# Basic dict comprehension
squares = {x: x**2 for x in range(6)}
# Result: {0: 0, 1: 1, 2: 4, 3: 9, 4: 16, 5: 25}

# Filtering
even_squares = {x: x**2 for x in range(10) if x % 2 == 0}

# Swapping keys and values
original = {'a': 1, 'b': 2, 'c': 3}
swapped = {v: k for k, v in original.items()}

# From two lists
keys = ['name', 'age', 'city']
values = ['Alice', 30, 'NYC']
person = {k: v for k, v in zip(keys, values)}
```

### Best Practice
Use for creating dictionaries from iterables efficiently.

---

## Topic: Generator Functions

### Description
Functions that yield values one at a time, saving memory.

### Example
```python
def fibonacci_generator(limit: int):
    """Generate Fibonacci numbers up to limit."""
    a, b = 0, 1
    while a < limit:
        yield a
        a, b = b, a + b

# Usage
for num in fibonacci_generator(100):
    print(num)

# Generator expression (similar to list comprehension)
squares_gen = (x**2 for x in range(1000000))  # Memory efficient
```

### Common Mistakes
- Using list when generator would save memory
- Trying to index a generator (must convert to list first)

### Best Practice
Use generators for large sequences to save memory.

---

## Topic: Lambda Functions

### Description
Anonymous functions for simple, one-line operations.

### Example
```python
# Basic lambda
square = lambda x: x ** 2

# With multiple arguments
add = lambda x, y: x + y

# Common use: sorting
students = [('Alice', 85), ('Bob', 92), ('Charlie', 78)]
sorted_by_grade = sorted(students, key=lambda s: s[1], reverse=True)

# With filter and map
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
evens = list(filter(lambda x: x % 2 == 0, numbers))
doubled = list(map(lambda x: x * 2, numbers))
```

### Common Mistakes
- Writing complex lambdas (use regular function instead)
- Storing lambdas in variables when def is clearer

### Best Practice
Use lambdas for simple, throwaway functions, especially with sort/filter/map.

---

## Topic: Decorators

### Description
Functions that modify the behavior of other functions.

### Example
```python
import functools
import time

def timer(func):
    """Decorator to measure function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)
    return "Done"

# Decorator with arguments
def repeat(times):
    """Decorator to repeat function execution."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)
def greet(name):
    print(f"Hello, {name}!")
```

### Best Practice
Always use `@functools.wraps` to preserve function metadata.

---

## Topic: Context Managers

### Description
Manage resources with guaranteed cleanup using `with` statement.

### Example
```python
from contextlib import contextmanager

# Class-based context manager
class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None

    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
        return False  # Don't suppress exceptions

# Function-based context manager
@contextmanager
def timer_context():
    """Context manager for timing code blocks."""
    import time
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"Elapsed: {end - start:.4f} seconds")

# Usage
with timer_context():
    # Code to time
    sum(range(1000000))
```

### Best Practice
Use context managers for any resource that needs cleanup.

---

## Topic: Default Dictionary

### Description
Dictionary that provides default values for missing keys.

### Example
```python
from collections import defaultdict

# Count occurrences
def count_words(text: str) -> dict:
    """Count word occurrences in text."""
    word_count = defaultdict(int)
    for word in text.lower().split():
        word_count[word] += 1
    return dict(word_count)

# Group items
def group_by_length(words: list) -> dict:
    """Group words by their length."""
    groups = defaultdict(list)
    for word in words:
        groups[len(word)].append(word)
    return dict(groups)

# Nested defaultdict
tree = lambda: defaultdict(tree)
taxonomy = tree()
taxonomy['Animal']['Mammal']['Dog'] = 'Canis familiaris'
```

### Best Practice
Use defaultdict to avoid KeyError and simplify counting/grouping.

---

## Topic: Named Tuples

### Description
Tuples with named fields for more readable code.

### Example
```python
from collections import namedtuple
from typing import NamedTuple

# Traditional namedtuple
Point = namedtuple('Point', ['x', 'y'])
p = Point(10, 20)
print(p.x, p.y)  # Access by name
print(p[0], p[1])  # Also works by index

# Typed NamedTuple (Python 3.6+)
class Person(NamedTuple):
    name: str
    age: int
    city: str = "Unknown"  # Default value

person = Person("Alice", 30)
print(f"{person.name} is {person.age} years old")
```

### Best Practice
Use NamedTuple for immutable data structures with named fields.

---

## Topic: Dataclasses

### Description
Classes primarily used to store data with less boilerplate.

### Example
```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class Student:
    name: str
    age: int
    grades: List[float] = field(default_factory=list)

    @property
    def average_grade(self) -> float:
        if not self.grades:
            return 0.0
        return sum(self.grades) / len(self.grades)

    def add_grade(self, grade: float) -> None:
        self.grades.append(grade)

# Frozen dataclass (immutable)
@dataclass(frozen=True)
class ImmutablePoint:
    x: float
    y: float

# Usage
student = Student("Bob", 20)
student.add_grade(85.5)
student.add_grade(92.0)
print(f"Average: {student.average_grade}")
```

### Best Practice
Use dataclasses for data containers; use regular classes for complex behavior.

---

## Topic: Enumerate Pattern

### Description
Get both index and value when iterating.

### Example
```python
# Basic enumerate
fruits = ['apple', 'banana', 'cherry']
for i, fruit in enumerate(fruits):
    print(f"{i}: {fruit}")

# Custom start index
for i, fruit in enumerate(fruits, start=1):
    print(f"{i}. {fruit}")

# With list comprehension
indexed_fruits = [(i, fruit) for i, fruit in enumerate(fruits)]
```

### Best Practice
Use enumerate instead of range(len(list)) for cleaner code.

---

## Topic: Zip Pattern

### Description
Combine multiple iterables element-wise.

### Example
```python
# Basic zip
names = ['Alice', 'Bob', 'Charlie']
ages = [25, 30, 35]
cities = ['NYC', 'LA', 'Chicago']

for name, age, city in zip(names, ages, cities):
    print(f"{name}, {age}, lives in {city}")

# Create dictionary from two lists
person_dict = dict(zip(names, ages))

# Unzip (transpose)
pairs = [(1, 'a'), (2, 'b'), (3, 'c')]
numbers, letters = zip(*pairs)

# Zip longest (from itertools)
from itertools import zip_longest
short = [1, 2]
long = [1, 2, 3, 4]
combined = list(zip_longest(short, long, fillvalue=0))
```

### Best Practice
Use zip for parallel iteration; use zip_longest when lengths differ.

---

## Topic: Any and All

### Description
Check if any or all elements satisfy a condition.

### Example
```python
numbers = [2, 4, 6, 8, 10]

# Check if all are even
all_even = all(n % 2 == 0 for n in numbers)  # True

# Check if any is greater than 5
any_large = any(n > 5 for n in numbers)  # True

# Check if list has any truthy values
has_truthy = any([0, '', None, False, 1])  # True (1 is truthy)

# Check if all are truthy
all_truthy = all([1, 'hello', True, [1]])  # True
```

### Best Practice
Use any/all with generator expressions for short-circuit evaluation.

---

## Topic: Unpacking

### Description
Extract values from sequences and dictionaries.

### Example
```python
# Basic unpacking
a, b, c = [1, 2, 3]

# Extended unpacking (Python 3)
first, *middle, last = [1, 2, 3, 4, 5]
# first=1, middle=[2, 3, 4], last=5

# Swap values
a, b = b, a

# Ignore values
_, important, _ = (1, 2, 3)

# Dictionary unpacking
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

kwargs = {"name": "Alice", "greeting": "Hi"}
print(greet(**kwargs))  # Hi, Alice!

# Merge dictionaries (Python 3.9+)
dict1 = {'a': 1, 'b': 2}
dict2 = {'c': 3, 'd': 4}
merged = {**dict1, **dict2}
```

### Best Practice
Use unpacking for cleaner, more readable assignments.

---

## Topic: Walrus Operator

### Description
Assignment expression (Python 3.8+) for inline assignment.

### Example
```python
# Without walrus operator
line = input()
while line != 'quit':
    print(f"You entered: {line}")
    line = input()

# With walrus operator
while (line := input()) != 'quit':
    print(f"You entered: {line}")

# In list comprehension
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
filtered = [y for x in data if (y := x * 2) > 10]

# In conditional
if (n := len(data)) > 5:
    print(f"List has {n} elements")
```

### Best Practice
Use sparingly; clarity is more important than brevity.

---

## Topic: Method Chaining

### Description
Calling multiple methods in sequence on an object.

### Example
```python
# String method chaining
text = "  Hello, World!  "
processed = text.strip().lower().replace(",", "").split()
# Result: ['hello', 'world!']

# Custom class with method chaining
class QueryBuilder:
    def __init__(self):
        self.query = ""

    def select(self, fields: str) -> 'QueryBuilder':
        self.query += f"SELECT {fields} "
        return self

    def from_table(self, table: str) -> 'QueryBuilder':
        self.query += f"FROM {table} "
        return self

    def where(self, condition: str) -> 'QueryBuilder':
        self.query += f"WHERE {condition}"
        return self

    def build(self) -> str:
        return self.query.strip()

# Usage
query = (QueryBuilder()
         .select("*")
         .from_table("users")
         .where("age > 18")
         .build())
```

### Best Practice
Return `self` from methods to enable chaining.
