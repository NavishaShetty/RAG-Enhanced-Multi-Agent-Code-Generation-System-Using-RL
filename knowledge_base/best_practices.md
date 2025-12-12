# Python Best Practices and Code Quality

## Topic: PEP 8 Style Guidelines

### Description
Python's official style guide for readable code.

### Example
```python
# Good: lowercase with underscores for functions/variables
def calculate_total_price(items):
    total_amount = 0
    for item in items:
        total_amount += item.price
    return total_amount

# Good: CamelCase for classes
class ShoppingCart:
    def __init__(self):
        self.items = []

# Good: UPPERCASE for constants
MAX_RETRY_ATTEMPTS = 3
DEFAULT_TIMEOUT = 30

# Good: 4 spaces for indentation (not tabs)
if condition:
    do_something()
    if nested_condition:
        do_nested_thing()

# Good: spaces around operators
x = 5 + 3
result = function(arg1, arg2)

# Good: line length <= 79 characters (or 99 for modern projects)
# Bad: very_long_function_name_that_exceeds_the_limit(argument1, argument2, argument3)
# Good:
result = very_long_function_name(
    argument1,
    argument2,
    argument3
)
```

### Best Practice
Use a linter (flake8, pylint) to automatically check PEP 8 compliance.

---

## Topic: Type Hints

### Description
Add type information to improve code clarity and enable static analysis.

### Example
```python
from typing import List, Dict, Optional, Tuple, Union, Callable

# Basic type hints
def greet(name: str) -> str:
    return f"Hello, {name}!"

# Collections
def sum_numbers(numbers: List[int]) -> int:
    return sum(numbers)

# Optional (can be None)
def find_user(user_id: int) -> Optional[Dict[str, str]]:
    if user_id == 1:
        return {"name": "Alice", "email": "alice@example.com"}
    return None

# Union types (multiple possible types)
def process_input(value: Union[int, str]) -> str:
    return str(value)

# Type alias
UserId = int
UserData = Dict[str, Union[str, int]]

def get_user(user_id: UserId) -> UserData:
    return {"name": "Alice", "age": 30}

# Callable type hint
def apply_operation(
    numbers: List[int],
    operation: Callable[[int], int]
) -> List[int]:
    return [operation(n) for n in numbers]

# Generic types (Python 3.9+)
def first_element(items: list[int]) -> int | None:
    return items[0] if items else None
```

### Best Practice
Add type hints to function signatures; use mypy for static type checking.

---

## Topic: Docstrings

### Description
Document functions, classes, and modules with structured docstrings.

### Example
```python
def calculate_discount(price: float, discount_percent: float) -> float:
    """
    Calculate the discounted price.

    Args:
        price: Original price of the item (must be positive).
        discount_percent: Discount percentage (0-100).

    Returns:
        The price after applying the discount.

    Raises:
        ValueError: If price is negative or discount is out of range.

    Example:
        >>> calculate_discount(100.0, 20)
        80.0
    """
    if price < 0:
        raise ValueError("Price must be non-negative")
    if not 0 <= discount_percent <= 100:
        raise ValueError("Discount must be between 0 and 100")

    discount_amount = price * (discount_percent / 100)
    return price - discount_amount


class BankAccount:
    """
    A simple bank account class.

    Attributes:
        owner: Name of the account owner.
        balance: Current balance in the account.

    Example:
        >>> account = BankAccount("Alice", 100)
        >>> account.deposit(50)
        >>> account.balance
        150
    """

    def __init__(self, owner: str, initial_balance: float = 0):
        """Initialize account with owner name and optional balance."""
        self.owner = owner
        self.balance = initial_balance
```

### Best Practice
Use Google or NumPy docstring style consistently; include examples.

---

## Topic: DRY Principle (Don't Repeat Yourself)

### Description
Avoid code duplication by extracting common logic.

### Example
```python
# Bad: Repeated logic
def validate_email(email):
    if not email:
        raise ValueError("Email cannot be empty")
    if "@" not in email:
        raise ValueError("Email must contain @")
    return email.lower().strip()

def validate_username(username):
    if not username:
        raise ValueError("Username cannot be empty")
    if len(username) < 3:
        raise ValueError("Username must be at least 3 characters")
    return username.lower().strip()

# Good: Extract common validation
def validate_required_string(value: str, field_name: str, min_length: int = 1) -> str:
    """Validate a required string field."""
    if not value:
        raise ValueError(f"{field_name} cannot be empty")
    if len(value) < min_length:
        raise ValueError(f"{field_name} must be at least {min_length} characters")
    return value.strip()

def validate_email(email: str) -> str:
    email = validate_required_string(email, "Email")
    if "@" not in email:
        raise ValueError("Email must contain @")
    return email.lower()

def validate_username(username: str) -> str:
    return validate_required_string(username, "Username", min_length=3).lower()
```

### Best Practice
Extract repeated logic into functions; use inheritance for shared behavior.

---

## Topic: Single Responsibility Principle

### Description
Each function/class should do one thing well.

### Example
```python
# Bad: Function does too much
def process_user_data(user_dict):
    # Validate
    if not user_dict.get('email'):
        raise ValueError("Email required")
    # Transform
    user_dict['email'] = user_dict['email'].lower()
    # Save to database
    db.save(user_dict)
    # Send email
    send_welcome_email(user_dict['email'])
    return user_dict

# Good: Separate responsibilities
def validate_user(user_dict: dict) -> None:
    """Validate user data."""
    if not user_dict.get('email'):
        raise ValueError("Email required")

def normalize_user(user_dict: dict) -> dict:
    """Normalize user data."""
    return {
        **user_dict,
        'email': user_dict['email'].lower()
    }

def save_user(user_dict: dict) -> None:
    """Save user to database."""
    db.save(user_dict)

def onboard_user(user_dict: dict) -> dict:
    """Complete user onboarding process."""
    validate_user(user_dict)
    user = normalize_user(user_dict)
    save_user(user)
    send_welcome_email(user['email'])
    return user
```

### Best Practice
If a function description includes "and", consider splitting it.

---

## Topic: Guard Clauses

### Description
Handle edge cases early to reduce nesting.

### Example
```python
# Bad: Deep nesting
def calculate_price(user, product, quantity):
    if user:
        if user.is_active:
            if product:
                if product.in_stock:
                    if quantity > 0:
                        base_price = product.price * quantity
                        if user.is_premium:
                            return base_price * 0.9
                        return base_price
    return None

# Good: Guard clauses
def calculate_price(user, product, quantity):
    """Calculate total price with validation."""
    if not user:
        raise ValueError("User required")
    if not user.is_active:
        raise ValueError("User account is not active")
    if not product:
        raise ValueError("Product required")
    if not product.in_stock:
        raise ValueError("Product is out of stock")
    if quantity <= 0:
        raise ValueError("Quantity must be positive")

    base_price = product.price * quantity
    discount = 0.9 if user.is_premium else 1.0
    return base_price * discount
```

### Best Practice
Return or raise early; keep the main logic at minimal indentation.

---

## Topic: Meaningful Names

### Description
Use descriptive, intention-revealing names.

### Example
```python
# Bad: Unclear names
def calc(a, b, c):
    return a * b * (1 - c)

x = calc(100, 5, 0.2)

# Good: Descriptive names
def calculate_discounted_total(
    unit_price: float,
    quantity: int,
    discount_rate: float
) -> float:
    """Calculate total price after applying discount."""
    subtotal = unit_price * quantity
    discount_amount = subtotal * discount_rate
    return subtotal - discount_amount

final_price = calculate_discounted_total(
    unit_price=100,
    quantity=5,
    discount_rate=0.2
)

# Good: Boolean names as questions
is_valid = True
has_permission = False
can_edit = user.role == 'admin'

# Good: Collection names as plurals
users = get_all_users()
active_orders = [o for o in orders if o.status == 'active']
```

### Best Practice
Names should explain WHAT the code does, not HOW.

---

## Topic: EAFP vs LBYL

### Description
"Easier to Ask Forgiveness than Permission" vs "Look Before You Leap"

### Example
```python
# LBYL (Look Before You Leap) - Not Pythonic
def get_value_lbyl(dictionary, key):
    if key in dictionary:
        return dictionary[key]
    else:
        return None

# EAFP (Easier to Ask Forgiveness than Permission) - Pythonic
def get_value_eafp(dictionary, key):
    try:
        return dictionary[key]
    except KeyError:
        return None

# Best: Use dict.get() for this specific case
def get_value_best(dictionary, key):
    return dictionary.get(key)

# LBYL - Checking file exists before opening
import os
if os.path.exists('config.json'):
    with open('config.json') as f:
        config = json.load(f)

# EAFP - Just try to open it
try:
    with open('config.json') as f:
        config = json.load(f)
except FileNotFoundError:
    config = {}
```

### Best Practice
Python prefers EAFP; use try/except for expected exceptions.

---

## Topic: Immutability and Pure Functions

### Description
Prefer immutable data and functions without side effects.

### Example
```python
# Bad: Mutable default argument
def add_item(item, items=[]):  # BUG! Default list is shared
    items.append(item)
    return items

# Good: Use None as default
def add_item(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items

# Better: Return new list (pure function)
def add_item_pure(item, items=None):
    """Add item to list without modifying original."""
    items = items or []
    return items + [item]  # Creates new list

# Bad: Modifying input
def normalize_users(users):
    for user in users:
        user['name'] = user['name'].title()  # Modifies original!
    return users

# Good: Create new data
def normalize_users_pure(users):
    """Return new list with normalized user data."""
    return [
        {**user, 'name': user['name'].title()}
        for user in users
    ]
```

### Best Practice
Avoid mutable default arguments; prefer returning new data over modifying input.

---

## Topic: Context Managers for Resources

### Description
Always use context managers for resource management.

### Example
```python
# Bad: Manual resource management
file = open('data.txt', 'r')
try:
    data = file.read()
finally:
    file.close()

# Good: Context manager
with open('data.txt', 'r') as file:
    data = file.read()

# Database connection
import sqlite3

# Bad
conn = sqlite3.connect('database.db')
cursor = conn.cursor()
cursor.execute('SELECT * FROM users')
results = cursor.fetchall()
conn.close()

# Good
with sqlite3.connect('database.db') as conn:
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users')
    results = cursor.fetchall()
# Connection automatically closed

# Multiple resources
with open('input.txt') as infile, open('output.txt', 'w') as outfile:
    outfile.write(infile.read().upper())
```

### Best Practice
Use `with` statement for files, connections, locks, and any resource needing cleanup.

---

## Topic: List vs Generator for Large Data

### Description
Use generators for memory efficiency with large datasets.

### Example
```python
# Bad: Loading all into memory
def read_large_file_bad(filename):
    with open(filename) as f:
        return f.readlines()  # All lines in memory

# Good: Generator for large files
def read_large_file(filename):
    with open(filename) as f:
        for line in f:
            yield line.strip()

# Bad: List comprehension for large data
squares = [x**2 for x in range(10_000_000)]  # Uses lots of memory

# Good: Generator expression
squares = (x**2 for x in range(10_000_000))  # Memory efficient

# Process without loading all
def process_data(filename):
    total = 0
    for line in read_large_file(filename):
        total += len(line)
    return total
```

### Best Practice
Use generators when you only need to iterate once through large data.

---

## Topic: Explicit is Better Than Implicit

### Description
Write clear, explicit code rather than clever, implicit code.

### Example
```python
# Bad: Implicit boolean conversion
def has_items(container):
    return bool(container)  # What types does this accept?

# Good: Explicit check
def has_items(items: list) -> bool:
    return len(items) > 0

# Bad: Implicit string formatting
name = "Alice"
greeting = "Hello, " + name + "!"

# Good: Explicit f-string
greeting = f"Hello, {name}!"

# Bad: Implicit type coercion
def add(a, b):
    return a + b  # Works with strings, numbers, lists...

# Good: Explicit types
def add_numbers(a: float, b: float) -> float:
    return a + b

# Bad: Relying on truthiness
if data:  # Empty list? None? Empty string?
    process(data)

# Good: Explicit check
if data is not None:
    process(data)
```

### Best Practice
Make intent clear; future readers (including yourself) will thank you.
