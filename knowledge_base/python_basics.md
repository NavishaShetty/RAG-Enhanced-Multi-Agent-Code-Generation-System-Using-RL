# Python Basics Knowledge Base

## Topic: String Reversal

### Description
Reversing a string in Python can be done efficiently using slicing.

### Example
```python
def reverse_string(s: str) -> str:
    """Reverse a string using slicing."""
    return s[::-1]
```

### Common Mistakes
- Using a loop to reverse character by character (inefficient)
- Forgetting that strings are immutable

### Best Practice
Use slice notation `[::-1]` for simplicity and performance.

---

## Topic: List Operations - Finding Maximum

### Description
Finding the maximum value in a list can be done with built-in functions or manually.

### Example
```python
def find_max(numbers: list) -> int:
    """Find the maximum value in a list."""
    if not numbers:
        raise ValueError("List cannot be empty")
    return max(numbers)

# Manual implementation for educational purposes
def find_max_manual(numbers: list) -> int:
    """Find maximum without using built-in max()."""
    if not numbers:
        raise ValueError("List cannot be empty")
    max_val = numbers[0]
    for num in numbers[1:]:
        if num > max_val:
            max_val = num
    return max_val
```

### Common Mistakes
- Not handling empty list case
- Using `sorted()` which is O(n log n) instead of `max()` which is O(n)

### Best Practice
Use `max()` for production code, but understand the manual approach.

---

## Topic: Sum of List Elements

### Description
Calculate the sum of all elements in a list.

### Example
```python
def sum_list(numbers: list) -> int:
    """Calculate sum of all numbers in a list."""
    return sum(numbers)

# Manual implementation
def sum_list_manual(numbers: list) -> int:
    """Calculate sum without using built-in sum()."""
    total = 0
    for num in numbers:
        total += num
    return total
```

### Common Mistakes
- Not handling empty lists (sum of empty list is 0, which is correct)
- Modifying the original list

### Best Practice
Use built-in `sum()` function for clarity and performance.

---

## Topic: Checking Even Numbers

### Description
Determine if a number is even using the modulo operator.

### Example
```python
def is_even(n: int) -> bool:
    """Check if a number is even."""
    return n % 2 == 0

def filter_even(numbers: list) -> list:
    """Filter and return only even numbers from a list."""
    return [n for n in numbers if n % 2 == 0]
```

### Common Mistakes
- Using division instead of modulo
- Not considering negative numbers (they work correctly with %)

### Best Practice
Use `n % 2 == 0` for checking even, or `n % 2 != 0` for odd.

---

## Topic: Counting Vowels

### Description
Count the number of vowels in a string.

### Example
```python
def count_vowels(s: str) -> int:
    """Count vowels in a string (case-insensitive)."""
    vowels = set('aeiouAEIOU')
    return sum(1 for char in s if char in vowels)

# Alternative implementation
def count_vowels_alt(s: str) -> int:
    """Count vowels using lowercase conversion."""
    vowels = 'aeiou'
    return sum(1 for char in s.lower() if char in vowels)
```

### Common Mistakes
- Forgetting case sensitivity
- Using a list instead of set for vowel lookup (less efficient)

### Best Practice
Use a set for O(1) lookup and handle both cases.

---

## Topic: Palindrome Check

### Description
Check if a string reads the same forwards and backwards.

### Example
```python
def is_palindrome(s: str) -> bool:
    """Check if string is a palindrome (case-insensitive, alphanumeric only)."""
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]

# Simple version for clean strings
def is_palindrome_simple(s: str) -> bool:
    """Check if string is a palindrome (simple version)."""
    return s == s[::-1]
```

### Common Mistakes
- Not handling spaces and punctuation
- Case sensitivity issues

### Best Practice
Clean the string first, then compare with its reverse.

---

## Topic: Factorial Calculation

### Description
Calculate the factorial of a non-negative integer.

### Example
```python
def factorial(n: int) -> int:
    """Calculate factorial recursively."""
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers")
    if n <= 1:
        return 1
    return n * factorial(n - 1)

# Iterative version (more efficient)
def factorial_iterative(n: int) -> int:
    """Calculate factorial iteratively."""
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
```

### Common Mistakes
- Not handling n=0 (0! = 1)
- Not handling negative numbers
- Stack overflow with large n in recursive version

### Best Practice
Use iterative version for large numbers, or use `math.factorial()`.

---

## Topic: Fibonacci Sequence

### Description
Generate Fibonacci numbers where each number is the sum of the two preceding ones.

### Example
```python
def fibonacci(n: int) -> int:
    """Return the nth Fibonacci number (0-indexed)."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def fibonacci_sequence(n: int) -> list:
    """Generate first n Fibonacci numbers."""
    if n <= 0:
        return []
    if n == 1:
        return [0]
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib
```

### Common Mistakes
- Off-by-one errors in indexing
- Using naive recursion (exponential time complexity)

### Best Practice
Use iterative approach with O(n) time and O(1) space.

---

## Topic: Finding Duplicates

### Description
Find duplicate elements in a list.

### Example
```python
def find_duplicates(lst: list) -> list:
    """Find all duplicate elements in a list."""
    seen = set()
    duplicates = set()
    for item in lst:
        if item in seen:
            duplicates.add(item)
        seen.add(item)
    return list(duplicates)

def has_duplicates(lst: list) -> bool:
    """Check if list contains any duplicates."""
    return len(lst) != len(set(lst))
```

### Common Mistakes
- Using nested loops (O(n^2) instead of O(n))
- Returning duplicate multiple times

### Best Practice
Use sets for O(n) time complexity.

---

## Topic: Binary Search

### Description
Efficiently search for an element in a sorted list.

### Example
```python
def binary_search(arr: list, target: int) -> int:
    """
    Binary search for target in sorted array.
    Returns index if found, -1 otherwise.
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
```

### Common Mistakes
- Integer overflow with `(left + right)` (use `left + (right - left) // 2` in other languages)
- Wrong boundary conditions (< vs <=)
- Not handling empty array

### Best Practice
Use `left <= right` condition and update boundaries correctly.

---

## Topic: Prime Number Check

### Description
Determine if a number is prime (divisible only by 1 and itself).

### Example
```python
def is_prime(n: int) -> bool:
    """Check if n is a prime number."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n ** 0.5) + 1, 2):
        if n % i == 0:
            return False
    return True
```

### Common Mistakes
- Not handling n < 2
- Checking all numbers up to n (inefficient)
- Forgetting to check even numbers separately

### Best Practice
Only check up to sqrt(n) and skip even numbers after 2.

---

## Topic: FizzBuzz

### Description
Classic programming problem: print numbers 1-n with substitutions.

### Example
```python
def fizzbuzz(n: int) -> list:
    """
    Generate FizzBuzz sequence from 1 to n.
    - Multiples of 3: 'Fizz'
    - Multiples of 5: 'Buzz'
    - Multiples of both: 'FizzBuzz'
    """
    result = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            result.append('FizzBuzz')
        elif i % 3 == 0:
            result.append('Fizz')
        elif i % 5 == 0:
            result.append('Buzz')
        else:
            result.append(str(i))
    return result
```

### Common Mistakes
- Checking 3 and 5 before 15
- Using string concatenation instead of checking 15

### Best Practice
Check 15 first, or build string incrementally.

---

## Topic: Sorting Algorithms - Bubble Sort

### Description
Simple comparison-based sorting algorithm.

### Example
```python
def bubble_sort(arr: list) -> list:
    """Sort list using bubble sort algorithm."""
    arr = arr.copy()  # Don't modify original
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break  # Already sorted
    return arr
```

### Common Mistakes
- Modifying the original list
- Not using early exit optimization

### Best Practice
Use built-in `sorted()` in production; bubble sort is O(n^2).

---

## Topic: Two Sum Problem

### Description
Find two numbers in a list that add up to a target sum.

### Example
```python
def two_sum(nums: list, target: int) -> list:
    """
    Find indices of two numbers that add to target.
    Returns [i, j] or [] if not found.
    """
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
```

### Common Mistakes
- Using nested loops (O(n^2))
- Returning numbers instead of indices

### Best Practice
Use a hash map for O(n) time complexity.
