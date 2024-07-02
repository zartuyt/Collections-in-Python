# Collections in Python

This project provides a set of abstract collection classes and their implementations in Python. These collections include `List`, `Stack`, `Queue`, and `Deque` with various implementations such as `LinkedList`, `DoubleLinkedList`, `ArrayList`, `ArrayListStack`, `LinkedListStack`, `LinkedListQueue`, and `CircularArrayDeque`.

## Abstract Classes

### `Collection`
An abstract base class for collections.
- `is_empty() -> bool`
- `empty() -> None`
- `size() -> int`
- `contains(e: object) -> bool`

### `List` (inherits from `Collection`)
An abstract base class for list collections.
- `add_first(e: object) -> None`
- `remove_first() -> None`
- `add_last(e: object) -> None`
- `remove_last() -> None`
- `first() -> object`
- `last() -> object`
- `replace(e: object, r: object) -> bool`
- `add_at(e: object, index: int) -> None`
- `get_at(index: int) -> object`
- `remove_at(index: int) -> None`

### `Stack` (inherits from `Collection`)
An abstract base class for stack collections.
- `push(e: object) -> None`
- `pop() -> object`
- `top() -> object`

### `Queue` (inherits from `Collection`)
An abstract base class for queue collections.
- `enqueue(e: object) -> None`
- `dequeue() -> object`
- `front() -> object`

### `Deque` (inherits from `Queue`)
An abstract base class for deque collections.
- `left_enqueue(e: object) -> None`
- `right_dequeue() -> object`

## Implementations

### `LinkedList`
A singly linked list implementation of the `List` abstract class.

### `DoubleLinkedList`
A doubly linked list implementation of the `List` abstract class.

### `ArrayList`
An array-based implementation of the `List` abstract class with dynamic resizing.

### `ArrayListStack`
A stack implementation using `ArrayList`.

### `LinkedListStack`
A stack implementation using `LinkedList`.

### `LinkedListQueue`
A queue implementation using `LinkedList`.

### `CircularArrayDeque`
A deque implementation using a circular array with dynamic resizing.

## Recursive Contains Function

### `recursive_contains(el: object, s: Stack) -> bool`
A function to check if an element exists in a stack using recursion.

## Custom Methods

### `SingleLinkedList` (inherits from `LinkedList`)
- `add_before(el: object, n: object) -> None`
- `add_before_recursive(el: object, n: object) -> None`
- `reorder() -> None`

### `LinkedListStack` (inherits from `LinkedList`)
- `add_after(el: object, n: object) -> None`
- `add_after_recursive(el: object, n: object) -> None`

## Unit Tests

Unit tests are provided for all implemented collections and methods using the `unittest` framework. To run the tests, execute the following command:

```bash
python -m unittest <test_module>.py
