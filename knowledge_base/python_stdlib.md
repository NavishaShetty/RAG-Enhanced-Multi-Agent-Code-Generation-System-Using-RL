# Python Standard Library Reference

## Topic: Collections Module - Counter

### Description
Count occurrences of elements in an iterable.

### Example
```python
from collections import Counter

# Count characters
char_count = Counter("mississippi")
# Counter({'i': 4, 's': 4, 'p': 2, 'm': 1})

# Count words
words = ['apple', 'banana', 'apple', 'cherry', 'banana', 'apple']
word_count = Counter(words)

# Most common elements
top_3 = word_count.most_common(3)  # [('apple', 3), ('banana', 2), ('cherry', 1)]

# Arithmetic operations
counter1 = Counter(a=3, b=1)
counter2 = Counter(a=1, b=2)
combined = counter1 + counter2  # Counter({'a': 4, 'b': 3})

# Elements iterator
list(Counter(a=2, b=3).elements())  # ['a', 'a', 'b', 'b', 'b']
```

### Best Practice
Use Counter for frequency counting instead of manual dictionary.

---

## Topic: Collections Module - deque

### Description
Double-ended queue for efficient append/pop from both ends.

### Example
```python
from collections import deque

# Create deque
d = deque([1, 2, 3])

# Operations on both ends
d.append(4)       # Add to right: [1, 2, 3, 4]
d.appendleft(0)   # Add to left: [0, 1, 2, 3, 4]
d.pop()           # Remove from right: 4
d.popleft()       # Remove from left: 0

# Rotate
d = deque([1, 2, 3, 4, 5])
d.rotate(2)   # [4, 5, 1, 2, 3]
d.rotate(-2)  # [1, 2, 3, 4, 5]

# Limited size deque (keeps last n items)
recent = deque(maxlen=3)
for i in range(5):
    recent.append(i)
# deque([2, 3, 4], maxlen=3)
```

### Best Practice
Use deque for FIFO queues or when you need O(1) operations on both ends.

---

## Topic: Itertools Module - Combinations and Permutations

### Description
Generate combinations and permutations of elements.

### Example
```python
from itertools import combinations, permutations, product

# Combinations (order doesn't matter, no repetition)
items = ['a', 'b', 'c']
combs = list(combinations(items, 2))
# [('a', 'b'), ('a', 'c'), ('b', 'c')]

# Permutations (order matters)
perms = list(permutations(items, 2))
# [('a', 'b'), ('a', 'c'), ('b', 'a'), ('b', 'c'), ('c', 'a'), ('c', 'b')]

# Cartesian product
product_result = list(product([1, 2], ['a', 'b']))
# [(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b')]

# Combinations with replacement
from itertools import combinations_with_replacement
combs_rep = list(combinations_with_replacement([1, 2, 3], 2))
# [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]
```

### Best Practice
Use itertools for memory-efficient iteration over combinations.

---

## Topic: Itertools Module - Chain and Groupby

### Description
Chain iterables and group consecutive elements.

### Example
```python
from itertools import chain, groupby

# Chain multiple iterables
list1 = [1, 2, 3]
list2 = [4, 5, 6]
list3 = [7, 8, 9]
chained = list(chain(list1, list2, list3))
# [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Flatten nested list
nested = [[1, 2], [3, 4], [5, 6]]
flat = list(chain.from_iterable(nested))
# [1, 2, 3, 4, 5, 6]

# Group by key (data must be sorted by key first!)
data = [('a', 1), ('a', 2), ('b', 3), ('b', 4), ('c', 5)]
for key, group in groupby(data, key=lambda x: x[0]):
    print(f"{key}: {list(group)}")
# a: [('a', 1), ('a', 2)]
# b: [('b', 3), ('b', 4)]
# c: [('c', 5)]

# Group consecutive numbers
numbers = [1, 1, 1, 2, 2, 3, 3, 3, 3]
groups = [(k, len(list(g))) for k, g in groupby(numbers)]
# [(1, 3), (2, 2), (3, 4)]
```

### Best Practice
Always sort data before using groupby; it only groups consecutive elements.

---

## Topic: Functools Module - reduce and partial

### Description
Functional programming utilities for reduction and partial application.

### Example
```python
from functools import reduce, partial

# Reduce - cumulative operation
numbers = [1, 2, 3, 4, 5]
product = reduce(lambda x, y: x * y, numbers)  # 120

# Sum with initial value
total = reduce(lambda x, y: x + y, numbers, 100)  # 115

# Find maximum using reduce
maximum = reduce(lambda x, y: x if x > y else y, numbers)

# Partial function application
def power(base, exponent):
    return base ** exponent

square = partial(power, exponent=2)
cube = partial(power, exponent=3)

print(square(5))  # 25
print(cube(5))    # 125

# Partial with positional args
def greet(greeting, name, punctuation):
    return f"{greeting}, {name}{punctuation}"

say_hello = partial(greet, "Hello")
print(say_hello("Alice", "!"))  # Hello, Alice!
```

### Best Practice
Use reduce sparingly; often a loop or built-in is clearer.

---

## Topic: Functools Module - lru_cache

### Description
Memoization decorator for caching function results.

### Example
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def fibonacci(n: int) -> int:
    """Cached Fibonacci calculation."""
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# Now fibonacci(100) is fast due to caching
print(fibonacci(100))

# Cache info
print(fibonacci.cache_info())
# CacheInfo(hits=98, misses=101, maxsize=128, currsize=101)

# Clear cache
fibonacci.cache_clear()

# Unlimited cache
@lru_cache(maxsize=None)
def factorial(n):
    if n < 2:
        return 1
    return n * factorial(n - 1)
```

### Best Practice
Use lru_cache for expensive pure functions with hashable arguments.

---

## Topic: OS Module - Path Operations

### Description
Operating system interface for file and directory operations.

### Example
```python
import os

# Current working directory
cwd = os.getcwd()

# List directory contents
files = os.listdir('.')

# Check if path exists
exists = os.path.exists('/some/path')
is_file = os.path.isfile('/some/path')
is_dir = os.path.isdir('/some/path')

# Join paths (platform-independent)
path = os.path.join('folder', 'subfolder', 'file.txt')

# Split path
dirname, filename = os.path.split('/path/to/file.txt')
name, ext = os.path.splitext('file.txt')  # ('file', '.txt')

# Create directories
os.makedirs('path/to/directory', exist_ok=True)

# Walk directory tree
for root, dirs, files in os.walk('.'):
    for file in files:
        full_path = os.path.join(root, file)
        print(full_path)

# Environment variables
home = os.environ.get('HOME', '/default/path')
```

### Best Practice
Use os.path.join for platform-independent path handling.

---

## Topic: Pathlib Module (Modern Path Handling)

### Description
Object-oriented filesystem paths (Python 3.4+).

### Example
```python
from pathlib import Path

# Create path object
path = Path('/home/user/documents')

# Path operations
file_path = path / 'file.txt'  # Join with /
parent = file_path.parent      # /home/user/documents
name = file_path.name          # file.txt
stem = file_path.stem          # file
suffix = file_path.suffix      # .txt

# Check existence
if file_path.exists():
    if file_path.is_file():
        print("It's a file")
    elif file_path.is_dir():
        print("It's a directory")

# Read and write files
content = Path('file.txt').read_text()
Path('output.txt').write_text('Hello, World!')

# Iterate directory
for item in Path('.').iterdir():
    print(item)

# Glob pattern matching
for py_file in Path('.').glob('**/*.py'):  # Recursive
    print(py_file)

# Create directories
Path('new/nested/dir').mkdir(parents=True, exist_ok=True)
```

### Best Practice
Prefer pathlib over os.path for new code; it's more Pythonic.

---

## Topic: JSON Module

### Description
Encode and decode JSON data.

### Example
```python
import json

# Python to JSON string
data = {
    "name": "Alice",
    "age": 30,
    "cities": ["NYC", "LA"],
    "active": True,
    "score": None
}
json_str = json.dumps(data, indent=2)

# JSON string to Python
parsed = json.loads(json_str)

# Read from file
with open('data.json', 'r') as f:
    data = json.load(f)

# Write to file
with open('output.json', 'w') as f:
    json.dump(data, f, indent=2)

# Custom encoder for non-serializable objects
from datetime import datetime

class DateEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

data_with_date = {"timestamp": datetime.now()}
json_str = json.dumps(data_with_date, cls=DateEncoder)
```

### Best Practice
Use `indent` parameter for readable output; handle encoding errors.

---

## Topic: Regular Expressions (re module)

### Description
Pattern matching and text manipulation.

### Example
```python
import re

text = "Contact: john@email.com or jane@company.org"

# Search for pattern
match = re.search(r'\w+@\w+\.\w+', text)
if match:
    print(match.group())  # john@email.com

# Find all matches
emails = re.findall(r'\w+@\w+\.\w+', text)
# ['john@email.com', 'jane@company.org']

# Substitute
cleaned = re.sub(r'\d+', 'NUM', "Price: 100 dollars, Qty: 5")
# "Price: NUM dollars, Qty: NUM"

# Split by pattern
parts = re.split(r'[,;]\s*', "apple, banana; cherry, date")
# ['apple', 'banana', 'cherry', 'date']

# Compile for reuse
email_pattern = re.compile(r'^[\w.-]+@[\w.-]+\.\w+$')
is_valid = email_pattern.match('user@example.com') is not None

# Groups
pattern = r'(\w+)@(\w+)\.(\w+)'
match = re.search(pattern, 'user@example.com')
if match:
    username, domain, tld = match.groups()
```

### Best Practice
Use raw strings (r'...') for patterns; compile patterns used repeatedly.

---

## Topic: Datetime Module

### Description
Date and time manipulation.

### Example
```python
from datetime import datetime, date, timedelta

# Current date and time
now = datetime.now()
today = date.today()

# Create specific datetime
dt = datetime(2024, 12, 25, 10, 30, 0)

# Parse string to datetime
parsed = datetime.strptime("2024-12-25", "%Y-%m-%d")

# Format datetime to string
formatted = now.strftime("%Y-%m-%d %H:%M:%S")

# Date arithmetic
tomorrow = today + timedelta(days=1)
next_week = today + timedelta(weeks=1)
diff = datetime(2025, 1, 1) - now

# Components
year = now.year
month = now.month
day = now.day
hour = now.hour
weekday = now.weekday()  # 0=Monday, 6=Sunday

# Compare dates
if datetime.now() > datetime(2024, 1, 1):
    print("After 2024 start")
```

### Best Practice
Use ISO format (YYYY-MM-DD) for dates; consider timezone-aware datetimes.

---

## Topic: Random Module

### Description
Generate random numbers and make random selections.

### Example
```python
import random

# Random float [0.0, 1.0)
r = random.random()

# Random integer in range [a, b] inclusive
num = random.randint(1, 10)

# Random float in range
f = random.uniform(1.0, 10.0)

# Random choice from sequence
colors = ['red', 'green', 'blue']
choice = random.choice(colors)

# Multiple random choices (with replacement)
choices = random.choices(colors, k=5)

# Sample without replacement
sample = random.sample(range(100), 10)

# Shuffle list in place
deck = list(range(52))
random.shuffle(deck)

# Set seed for reproducibility
random.seed(42)
print(random.random())  # Always same result

# Random from normal distribution
value = random.gauss(mu=0, sigma=1)
```

### Best Practice
Set seed for reproducible results in testing; use secrets module for security.

---

## Topic: Math Module

### Description
Mathematical functions and constants.

### Example
```python
import math

# Constants
pi = math.pi        # 3.141592...
e = math.e          # 2.718281...
inf = math.inf      # Infinity
nan = math.nan      # Not a Number

# Basic functions
sqrt = math.sqrt(16)           # 4.0
power = math.pow(2, 10)        # 1024.0
abs_val = math.fabs(-5.5)      # 5.5

# Rounding
ceil = math.ceil(4.2)          # 5
floor = math.floor(4.8)        # 4
trunc = math.trunc(-4.8)       # -4

# Logarithms
log = math.log(math.e)         # 1.0 (natural log)
log10 = math.log10(100)        # 2.0
log2 = math.log2(8)            # 3.0

# Trigonometry
sin = math.sin(math.pi / 2)    # 1.0
cos = math.cos(0)              # 1.0
radians = math.radians(180)    # pi

# Other useful functions
factorial = math.factorial(5)   # 120
gcd = math.gcd(48, 18)         # 6
lcm = math.lcm(4, 6)           # 12 (Python 3.9+)
```

### Best Practice
Use math functions for numerical accuracy; they're implemented in C.
