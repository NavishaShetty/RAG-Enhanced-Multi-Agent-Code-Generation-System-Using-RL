# Python Error Handling and Exception Patterns

## Topic: Basic Try-Except

### Description
Catch and handle exceptions gracefully.

### Example
```python
# Basic exception handling
def divide(a: float, b: float) -> float:
    """Divide two numbers with error handling."""
    try:
        return a / b
    except ZeroDivisionError:
        print("Error: Cannot divide by zero")
        return 0.0

# Multiple exception types
def parse_int(value: str) -> int:
    """Parse string to integer safely."""
    try:
        return int(value)
    except ValueError:
        print(f"'{value}' is not a valid integer")
        return 0
    except TypeError:
        print(f"Expected string, got {type(value)}")
        return 0

# Catch multiple exceptions in one handler
def safe_convert(value):
    try:
        return int(value)
    except (ValueError, TypeError) as e:
        print(f"Conversion error: {e}")
        return None
```

### Best Practice
Catch specific exceptions; avoid bare `except:` clauses.

---

## Topic: Try-Except-Else-Finally

### Description
Complete exception handling with all clauses.

### Example
```python
def read_file_safely(filename: str) -> str:
    """Read file with complete error handling."""
    file = None
    try:
        file = open(filename, 'r')
        content = file.read()
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return ""
    except PermissionError:
        print(f"Permission denied: {filename}")
        return ""
    except IOError as e:
        print(f"IO error reading file: {e}")
        return ""
    else:
        # Runs only if no exception occurred
        print(f"Successfully read {len(content)} characters")
        return content
    finally:
        # Always runs, even if exception occurred
        if file:
            file.close()
            print("File closed")


# Common pattern: database transaction
def save_to_database(data):
    """Save data with transaction handling."""
    connection = get_database_connection()
    try:
        connection.begin_transaction()
        connection.insert(data)
    except DatabaseError as e:
        connection.rollback()
        raise  # Re-raise after cleanup
    else:
        connection.commit()
        print("Data saved successfully")
    finally:
        connection.close()
```

### Best Practice
Use `else` for code that should run only on success; `finally` for cleanup.

---

## Topic: Raising Exceptions

### Description
Raise appropriate exceptions with meaningful messages.

### Example
```python
def validate_age(age: int) -> int:
    """Validate age is within acceptable range."""
    if not isinstance(age, int):
        raise TypeError(f"Age must be integer, got {type(age).__name__}")
    if age < 0:
        raise ValueError("Age cannot be negative")
    if age > 150:
        raise ValueError("Age seems unrealistic (max: 150)")
    return age


def get_user(user_id: int):
    """Get user by ID or raise appropriate exception."""
    if not isinstance(user_id, int):
        raise TypeError("user_id must be an integer")
    if user_id < 0:
        raise ValueError("user_id must be positive")

    user = database.find_user(user_id)
    if user is None:
        raise LookupError(f"User with ID {user_id} not found")

    return user


# Re-raising with additional context
def process_config(filename: str):
    """Process configuration file."""
    try:
        with open(filename) as f:
            config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file missing: {filename}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {filename}: {e}")
    return config
```

### Best Practice
Raise the most specific exception type; include helpful error messages.

---

## Topic: Custom Exceptions

### Description
Create domain-specific exception classes.

### Example
```python
# Base exception for your application
class ApplicationError(Exception):
    """Base exception for application errors."""
    pass


class ValidationError(ApplicationError):
    """Raised when data validation fails."""

    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(f"Validation error on '{field}': {message}")


class AuthenticationError(ApplicationError):
    """Raised when authentication fails."""
    pass


class AuthorizationError(ApplicationError):
    """Raised when user lacks required permissions."""

    def __init__(self, user: str, action: str):
        self.user = user
        self.action = action
        super().__init__(f"User '{user}' not authorized for action '{action}'")


# Usage
def create_user(username: str, email: str):
    """Create a new user with validation."""
    if len(username) < 3:
        raise ValidationError("username", "Must be at least 3 characters")
    if "@" not in email:
        raise ValidationError("email", "Invalid email format")

    # Create user...
    return {"username": username, "email": email}


# Handling custom exceptions
try:
    user = create_user("ab", "invalid-email")
except ValidationError as e:
    print(f"Field: {e.field}, Message: {e.message}")
except ApplicationError as e:
    print(f"Application error: {e}")
```

### Best Practice
Create exception hierarchy for your application; inherit from Exception.

---

## Topic: Exception Chaining

### Description
Preserve original exception context when re-raising.

### Example
```python
class DataProcessingError(Exception):
    """Error during data processing."""
    pass


def parse_data(raw_data: str):
    """Parse raw data string."""
    try:
        return json.loads(raw_data)
    except json.JSONDecodeError as e:
        # Chain exceptions to preserve original traceback
        raise DataProcessingError(f"Failed to parse data") from e


def process_file(filename: str):
    """Process a data file."""
    try:
        with open(filename) as f:
            return parse_data(f.read())
    except FileNotFoundError as e:
        raise DataProcessingError(f"Data file not found: {filename}") from e
    except DataProcessingError:
        raise  # Re-raise as-is


# Suppress original exception (rare)
try:
    risky_operation()
except SomeError:
    raise DifferentError("Something went wrong") from None
```

### Best Practice
Use `from e` to chain exceptions; use `from None` to suppress original (rarely needed).

---

## Topic: Context Manager Exception Handling

### Description
Handle exceptions in context managers properly.

### Example
```python
from contextlib import contextmanager


@contextmanager
def managed_resource(name: str):
    """Context manager with exception handling."""
    print(f"Acquiring resource: {name}")
    resource = acquire_resource(name)
    try:
        yield resource
    except Exception as e:
        print(f"Error while using resource: {e}")
        resource.rollback()
        raise  # Re-raise after cleanup
    finally:
        print(f"Releasing resource: {name}")
        resource.release()


# Class-based context manager
class DatabaseTransaction:
    """Context manager for database transactions."""

    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        self.connection.begin()
        return self.connection

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Exception occurred, rollback
            self.connection.rollback()
            print(f"Transaction rolled back due to: {exc_val}")
            return False  # Don't suppress exception
        else:
            # Success, commit
            self.connection.commit()
            return True


# Usage
with DatabaseTransaction(conn) as db:
    db.execute("INSERT INTO users ...")
    db.execute("UPDATE accounts ...")
```

### Best Practice
Always clean up in `__exit__` or `finally`; return False to propagate exceptions.

---

## Topic: Assertions

### Description
Use assertions for internal consistency checks, not user input validation.

### Example
```python
# Good: Assert internal invariants
def binary_search(arr: list, target: int) -> int:
    """Binary search - array must be sorted."""
    assert arr == sorted(arr), "Array must be sorted"

    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        assert 0 <= mid < len(arr), "Mid index out of bounds"

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1


# Bad: Assertions for user input (can be disabled with -O flag)
def bad_validate_age(age):
    assert age >= 0, "Age must be positive"  # Don't do this!
    return age


# Good: Exceptions for user input
def good_validate_age(age: int) -> int:
    if age < 0:
        raise ValueError("Age must be positive")
    return age


# Assert for debugging
def complex_calculation(data):
    result = step_one(data)
    assert result is not None, "step_one should never return None"

    result = step_two(result)
    assert len(result) > 0, "step_two should produce non-empty result"

    return result
```

### Best Practice
Use assertions for debugging; use exceptions for runtime error handling.

---

## Topic: Logging Exceptions

### Description
Properly log exceptions for debugging and monitoring.

### Example
```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_data(data):
    """Process data with proper logging."""
    logger.info(f"Processing {len(data)} items")

    try:
        result = transform(data)
        logger.debug(f"Transform successful: {len(result)} items")
        return result

    except ValueError as e:
        logger.warning(f"Invalid data encountered: {e}")
        return []

    except Exception as e:
        # Log exception with full traceback
        logger.exception(f"Unexpected error processing data: {e}")
        raise


def api_call(endpoint: str):
    """Make API call with error logging."""
    try:
        response = requests.get(endpoint)
        response.raise_for_status()
        return response.json()

    except requests.HTTPError as e:
        logger.error(f"HTTP error {e.response.status_code}: {endpoint}")
        raise

    except requests.ConnectionError:
        logger.error(f"Connection failed: {endpoint}")
        raise

    except Exception:
        logger.exception(f"Unexpected error calling {endpoint}")
        raise
```

### Best Practice
Use `logger.exception()` to include traceback; use appropriate log levels.

---

## Topic: Retry Pattern

### Description
Retry failed operations with exponential backoff.

### Example
```python
import time
import random
from functools import wraps


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator to retry function on exception.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Multiplier for delay after each retry
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            current_delay = delay

            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts == max_attempts:
                        raise

                    # Add jitter to prevent thundering herd
                    jitter = random.uniform(0, 0.1 * current_delay)
                    sleep_time = current_delay + jitter

                    print(f"Attempt {attempts} failed: {e}")
                    print(f"Retrying in {sleep_time:.2f} seconds...")

                    time.sleep(sleep_time)
                    current_delay *= backoff

            raise RuntimeError("Max retries exceeded")  # Should not reach here
        return wrapper
    return decorator


@retry(max_attempts=3, delay=1.0, backoff=2.0)
def fetch_data(url: str):
    """Fetch data from URL with automatic retry."""
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    return response.json()
```

### Best Practice
Add exponential backoff and jitter; set maximum attempts; log retry attempts.

---

## Topic: Graceful Degradation

### Description
Provide fallback behavior when operations fail.

### Example
```python
from typing import Optional


def get_user_preference(user_id: int, key: str, default: str = "") -> str:
    """Get user preference with fallback to default."""
    try:
        # Try to get from cache first
        value = cache.get(f"pref:{user_id}:{key}")
        if value is not None:
            return value

        # Try database
        value = database.get_preference(user_id, key)
        if value is not None:
            cache.set(f"pref:{user_id}:{key}", value, ttl=3600)
            return value

    except CacheError:
        # Cache is down, continue without caching
        pass

    except DatabaseError:
        # Database is down, return default
        logger.warning(f"Database error getting preference for user {user_id}")

    return default


def fetch_with_fallback(primary_url: str, fallback_url: str) -> dict:
    """Fetch data with fallback to secondary source."""
    try:
        return fetch_data(primary_url)
    except Exception as e:
        logger.warning(f"Primary fetch failed: {e}, trying fallback")

    try:
        return fetch_data(fallback_url)
    except Exception as e:
        logger.error(f"Fallback fetch also failed: {e}")
        return {}  # Return empty dict as last resort


def calculate_with_timeout(func, timeout: float, default=None):
    """Run function with timeout, return default if it fails."""
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(func)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            logger.warning("Calculation timed out, using default")
            return default
```

### Best Practice
Always have a fallback; log failures for monitoring; don't hide all errors.

---

## Topic: Input Validation Pattern

### Description
Validate input at system boundaries.

### Example
```python
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of validation operation."""
    is_valid: bool
    errors: List[str]

    @classmethod
    def success(cls) -> 'ValidationResult':
        return cls(is_valid=True, errors=[])

    @classmethod
    def failure(cls, errors: List[str]) -> 'ValidationResult':
        return cls(is_valid=False, errors=errors)


def validate_user_input(data: dict) -> ValidationResult:
    """Validate user registration data."""
    errors = []

    # Required fields
    if not data.get('username'):
        errors.append("Username is required")
    elif len(data['username']) < 3:
        errors.append("Username must be at least 3 characters")
    elif len(data['username']) > 50:
        errors.append("Username must be at most 50 characters")

    if not data.get('email'):
        errors.append("Email is required")
    elif '@' not in data['email']:
        errors.append("Invalid email format")

    if not data.get('password'):
        errors.append("Password is required")
    elif len(data['password']) < 8:
        errors.append("Password must be at least 8 characters")

    # Optional field validation
    if 'age' in data:
        try:
            age = int(data['age'])
            if not 0 <= age <= 150:
                errors.append("Age must be between 0 and 150")
        except (ValueError, TypeError):
            errors.append("Age must be a number")

    return ValidationResult.failure(errors) if errors else ValidationResult.success()


# Usage in API endpoint
def register_user(request_data: dict):
    """Handle user registration request."""
    validation = validate_user_input(request_data)

    if not validation.is_valid:
        return {
            "success": False,
            "errors": validation.errors
        }

    # Proceed with validated data
    user = create_user(request_data)
    return {"success": True, "user_id": user.id}
```

### Best Practice
Validate early; return all errors at once; separate validation from business logic.
