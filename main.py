import unittest
from abc import ABC, abstractmethod

class Collection(ABC):
    @abstractmethod
    def is_empty(self) -> bool:
        pass

    @abstractmethod
    def empty(self) -> None:
        pass

    @abstractmethod
    def size(self) -> int:
        pass

    @abstractmethod
    def contains(self, e: object) -> bool:
        pass


class List(Collection):
    @abstractmethod
    def add_first(self, e: object) -> None:
        pass

    @abstractmethod
    def remove_first(self) -> None:
        pass

    @abstractmethod
    def add_last(self, e: object) -> None:
        pass

    @abstractmethod
    def remove_last(self) -> None:
        pass

    @abstractmethod
    def first(self) -> object:
        pass

    @abstractmethod
    def last(self) -> object:
        pass

    @abstractmethod
    def replace(self, e: object, r: object) -> bool:
        pass

    @abstractmethod
    def add_at(self, e: object, index: int) -> None:
        pass

    @abstractmethod
    def get_at(self, index: int) -> object:
        pass

    @abstractmethod
    def remove_at(self, index: int) -> None:
        pass


class Stack(Collection):
    @abstractmethod
    def push(self, e: object) -> None:
        pass

    @abstractmethod
    def pop(self) -> object:
        pass

    @abstractmethod
    def top(self) -> object:
        pass


class Queue(Collection):
    @abstractmethod
    def enqueue(self, e: object) -> None:
        pass

    @abstractmethod
    def dequeue(self) -> object:
        pass

    @abstractmethod
    def front(self) -> object:
        pass


class Deque(Queue):
    @abstractmethod
    def left_enqueue(self, e: object) -> None:
        pass

    @abstractmethod
    def right_dequeue(self) -> object:
        pass

class Node:
    def __init__(self, data=None):
        self.data = data
        self.next = None

class LinkedList(List):
    def __init__(self):
        self.head = None
        self._size = 0

    def is_empty(self) -> bool:
        return self.head is None

    def empty(self) -> None:
        self.head = None
        self._size = 0

    def size(self) -> int:
        return self._size

    def contains(self, e: object) -> bool:
        current = self.head
        while current:
            if current.data == e:
                return True
            current = current.next
        return False

    def add_first(self, e: object) -> None:
        new_node = Node(e)
        new_node.next = self.head
        self.head = new_node
        self._size += 1

    def remove_first(self) -> None:
        if self.head:
            self.head = self.head.next
            self._size -= 1

    def add_last(self, e: object) -> None:
        new_node = Node(e)
        if not self.head:
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
        self._size += 1

    def remove_last(self) -> None:
        if not self.head:
            return
        if not self.head.next:
            self.head = None
        else:
            current = self.head
            while current.next and current.next.next:
                current = current.next
            current.next = None
        self._size -= 1

    def first(self) -> object:
        if self.head:
            return self.head.data
        return None

    def last(self) -> object:
        if not self.head:
            return None
        current = self.head
        while current.next:
            current = current.next
        return current.data

    def replace(self, e: object, r: object) -> bool:
        current = self.head
        while current:
            if current.data == e:
                current.data = r
                return True
            current = current.next
        return False

    def add_at(self, e: object, index: int) -> None:
        if index < 0 or index > self._size:
            raise IndexError("Index out of bounds")
        if index == 0:
            self.add_first(e)
            return
        new_node = Node(e)
        current = self.head
        for _ in range(index - 1):
            current = current.next
        new_node.next = current.next
        current.next = new_node
        self._size += 1

    def get_at(self, index: int) -> object:
        if index < 0 or index >= self._size:
            raise IndexError("Index out of bounds")
        current = self.head
        for _ in range(index):
            current = current.next
        return current.data

    def remove_at(self, index: int) -> None:
        if index < 0 or index >= self._size:
            raise IndexError("Index out of bounds")
        if index == 0:
            self.remove_first()
            return
        current = self.head
        for _ in range(index - 1):
            current = current.next
        current.next = current.next.next
        self._size -= 1

class DoubleNode:
    def __init__(self, data=None):
        self.data = data
        self.next = None
        self.prev = None

class DoubleLinkedList(List):
    def __init__(self):
        self.head = None
        self.tail = None
        self._size = 0

    def is_empty(self) -> bool:
        return self.head is None

    def empty(self) -> None:
        self.head = None
        self.tail = None
        self._size = 0

    def size(self) -> int:
        return self._size

    def contains(self, e: object) -> bool:
        current = self.head
        while current:
            if current.data == e:
                return True
            current = current.next
        return False

    def add_first(self, e: object) -> None:
        new_node = DoubleNode(e)
        if not self.head:
            self.head = new_node
            self.tail = new_node
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
        self._size += 1

    def remove_first(self) -> None:
        if not self.head:
            return
        if self.head == self.tail:
            self.head = None
            self.tail = None
        else:
            self.head = self.head.next
            self.head.prev = None
        self._size -= 1

    def add_last(self, e: object) -> None:
        new_node = DoubleNode(e)
        if not self.tail:
            self.head = new_node
            self.tail = new_node
        else:
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node
        self._size += 1

    def remove_last(self) -> None:
        if not self.tail:
            return
        if self.head == self.tail:
            self.head = None
            self.tail = None
        else:
            self.tail = self.tail.prev
            self.tail.next = None
        self._size -= 1

    def first(self) -> object:
        if self.head:
            return self.head.data
        return None

    def last(self) -> object:
        if self.tail:
            return self.tail.data
        return None

    def replace(self, e: object, r: object) -> bool:
        current = self.head
        while current:
            if current.data == e:
                current.data = r
                return True
            current = current.next
        return False

    def add_at(self, e: object, index: int) -> None:
        if index < 0 or index > self._size:
            raise IndexError("Index out of bounds")
        if index == 0:
            self.add_first(e)
            return
        if index == self._size:
            self.add_last(e)
            return
        new_node = DoubleNode(e)
        current = self.head
        for _ in range(index - 1):
            current = current.next
        new_node.next = current.next
        new_node.prev = current
        current.next.prev = new_node
        current.next = new_node
        self._size += 1

    def get_at(self, index: int) -> object:
        if index < 0 or index >= self._size:
            raise IndexError("Index out of bounds")
        current = self.head
        for _ in range(index):
            current = current.next
        return current.data

    def remove_at(self, index: int) -> None:
        if index < 0 or index >= self._size:
            raise IndexError("Index out of bounds")
        if index == 0:
            self.remove_first()
            return
        if index == self._size - 1:
            self.remove_last()
            return
        current = self.head
        for _ in range(index - 1):
            current = current.next
        current.next = current.next.next
        current.next.prev = current
        self._size -= 1

class ArrayList(List):
    def __init__(self, capacity=10):
        self.array = [None] * capacity
        self._size = 0
        self._capacity = capacity

    def is_empty(self) -> bool:
        return self._size == 0

    def empty(self) -> None:
        self.array = [None] * self._capacity
        self._size = 0

    def size(self) -> int:
        return self._size

    def contains(self, e: object) -> bool:
        for i in range(self._size):
            if self.array[i] == e:
                return True
        return False

    def add_first(self, e: object) -> None:
        if self._size == self._capacity:
            self._resize()
        for i in range(self._size, 0, -1):
            self.array[i] = self.array[i - 1]
        self.array[0] = e
        self._size += 1

    def remove_first(self) -> None:
        if self._size == 0:
            return
        for i in range(1, self._size):
            self.array[i - 1] = self.array[i]
        self._size -= 1

    def add_last(self, e: object) -> None:
        if self._size == self._capacity:
            self._resize()
        self.array[self._size] = e
        self._size += 1

    def remove_last(self) -> None:
        if self._size == 0:
            return
        self._size -= 1

    def first(self) -> object:
        if self._size == 0:
            return None
        return self.array[0]

    def last(self) -> object:
        if self._size == 0:
            return None
        return self.array[self._size - 1]

    def replace(self, e: object, r: object) -> bool:
        for i in range(self._size):
            if self.array[i] == e:
                self.array[i] = r
                return True
        return False

    def add_at(self, e: object, index: int) -> None:
        if index < 0 or index > self._size:
            raise IndexError("Index out of bounds")
        if self._size == self._capacity:
            self._resize()
        for i in range(self._size, index, -1):
            self.array[i] = self.array[i - 1]
        self.array[index] = e
        self._size += 1

    def get_at(self, index: int) -> object:
        if index < 0 or index >= self._size:
            raise IndexError("Index out of bounds")
        return self.array[index]

    def remove_at(self, index: int) -> None:
        if index < 0 or index >= self._size:
            raise IndexError("Index out of bounds")
        for i in range(index, self._size - 1):
            self.array[i] = self.array[i + 1]
        self._size -= 1

    def _resize(self) -> None:
        self._capacity *= 2
        new_array = [None] * self._capacity
        for i in range(self._size):
            new_array[i] = self.array[i]
        self.array = new_array

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index < self._size:
            result = self.array[self._index]
            self._index += 1
            return result
        else:
            raise StopIteration


class ArrayListStack(Stack):
    def __init__(self):
        self.array_list = ArrayList()

    def is_empty(self) -> bool:
        return self.array_list.is_empty()

    def empty(self) -> None:
        self.array_list.empty()

    def size(self) -> int:
        return self.array_list.size()

    def contains(self, e: object) -> bool:
        return self.array_list.contains(e)

    def push(self, e: object) -> None:
        self.array_list.add_last(e)

    def pop(self) -> object:
        if self.is_empty():
            raise IndexError("Pop from empty stack")
        top_element = self.array_list.last()
        self.array_list.remove_last()
        return top_element

    def top(self) -> object:
        if self.is_empty():
            raise IndexError("Top from empty stack")
        return self.array_list.last()

    def __iter__(self):
        return iter(self.array_list)



class LinkedListQueue(Queue):
    def __init__(self):
        self.linked_list = LinkedList()

    def is_empty(self) -> bool:
        return self.linked_list.is_empty()

    def empty(self) -> None:
        self.linked_list.empty()

    def size(self) -> int:
        return self.linked_list.size()

    def contains(self, e: object) -> bool:
        return self.linked_list.contains(e)

    def enqueue(self, e: object) -> None:
        self.linked_list.add_last(e)

    def dequeue(self) -> object:
        if self.is_empty():
            raise IndexError("Dequeue from empty queue")
        front_element = self.linked_list.first()
        self.linked_list.remove_first()
        return front_element

    def front(self) -> object:
        if self.is_empty():
            raise IndexError("Front from empty queue")
        return self.linked_list.first()

class CircularArrayDeque(Deque):
    def __init__(self, capacity=10):
        self.array = [None] * capacity
        self.front_index = 0
        self.back_index = 0
        self._size = 0
        self._capacity = capacity

    def is_empty(self) -> bool:
        return self._size == 0

    def empty(self) -> None:
        self.array = [None] * self._capacity
        self.front_index = 0
        self.back_index = 0
        self._size = 0

    def size(self) -> int:
        return self._size

    def contains(self, e: object) -> bool:
        for i in range(self._size):
            if self.array[(self.front_index + i) % self._capacity] == e:
                return True
        return False

    def enqueue(self, e: object) -> None:
        if self._size == self._capacity:
            self._resize()
        self.array[self.back_index] = e
        self.back_index = (self.back_index + 1) % self._capacity
        self._size += 1

    def dequeue(self) -> object:
        if self.is_empty():
            raise IndexError("Dequeue from empty deque")
        front_element = self.array[self.front_index]
        self.array[self.front_index] = None
        self.front_index = (self.front_index + 1) % self._capacity
        self._size -= 1
        return front_element

    def front(self) -> object:
        if self.is_empty():
            raise IndexError("Front from empty deque")
        return self.array[self.front_index]

    def left_enqueue(self, e: object) -> None:
        if self._size == self._capacity:
            self._resize()
        self.front_index = (self.front_index - 1 + self._capacity) % self._capacity
        self.array[self.front_index] = e
        self._size += 1

    def right_dequeue(self) -> object:
        if self.is_empty():
            raise IndexError("Dequeue from empty deque")
        self.back_index = (self.back_index - 1 + self._capacity) % self._capacity
        back_element = self.array[self.back_index]
        self.array[self.back_index] = None
        self._size -= 1
        return back_element

    def _resize(self) -> None:
        new_capacity = self._capacity * 2
        new_array = [None] * new_capacity
        for i in range(self._size):
            new_array[i] = self.array[(self.front_index + i) % self._capacity]
        self.array = new_array
        self.front_index = 0
        self.back_index = self._size
        self._capacity = new_capacity

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index < self._size:
            result = self.array[(self.front_index + self._index) % self._capacity]
            self._index += 1
            return result
        else:
            raise StopIteration

    class DivisibleByIterator:
        def __init__(self, array, front_index, size, capacity, divisor):
            self.array = array
            self.index = 0
            self.front_index = front_index
            self.size = size
            self.capacity = capacity
            self.divisor = divisor

        def __iter__(self):
            return self

        def __next__(self):
            while self.index < self.size:
                value = self.array[(self.front_index + self.index) % self.capacity]
                self.index += 1
                if value % self.divisor == 0:
                    return value
            raise StopIteration

    def divisible_by(self, divisor):
        return self.DivisibleByIterator(self.array, self.front_index, self.size, self.capacity, divisor)

def recursive_contains(el: object, s: Stack) -> bool:
    if s.is_empty():
        return False
    top_element = s.pop()
    if top_element == el:
        s.push(top_element)
        return True
    result = recursive_contains(el, s)
    s.push(top_element)
    return result


class SingleLinkedList(LinkedList):
    def add_before(self, el: object, n: object) -> None:
        # Iterative
        if self.is_empty():
            return
        if self.head.data == el:
            self.add_first(n)
            return
        current = self.head
        while current.next:
            if current.next.data == el:
                new_node = Node(n)
                new_node.next = current.next
                current.next = new_node
                self._size += 1
                return
            current = current.next

    def add_before_recursive(self, el: object, n: object) -> None:
        def _add_before_recursive(node, el, n):
            if node.next and node.next.data == el:
                new_node = Node(n)
                new_node.next = node.next
                node.next = new_node
                self._size += 1
                return
            if node.next:
                _add_before_recursive(node.next, el, n)

        if self.is_empty():
            return
        if self.head.data == el:
            self.add_first(n)
            return
        _add_before_recursive(self.head, el, n)
    def reorder(self) -> None:
        if self._size < 2:
            return

        slow = self.head
        fast = self.head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        prev = None
        current = slow
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node

        first = self.head
        second = prev
        while second.next:
            tmp1 = first.next
            tmp2 = second.next
            first.next = second
            second.next = tmp1
            first = tmp1
            second = tmp2

class LinkedListStack(Stack):
    def __init__(self):
        self.linked_list = LinkedList()

    def is_empty(self) -> bool:
        return self.linked_list.is_empty()

    def empty(self) -> None:
        self.linked_list.empty()

    def size(self) -> int:
        return self.linked_list.size()

    def contains(self, e: object) -> bool:
        return self.linked_list.contains(e)

    def push(self, e: object) -> None:
        self.linked_list.add_first(e)

    def pop(self) -> object:
        if self.is_empty():
            raise IndexError("Pop from empty stack")
        top_element = self.linked_list.first()
        self.linked_list.remove_first()
        return top_element

    def top(self) -> object:
        if self.is_empty():
            raise IndexError("Top from empty stack")
        return self.linked_list.first()

    def add_after(self, el: object, n: object) -> None:
        current = self.linked_list.head
        while current:
            if current.data == el:
                new_node = Node(n)
                new_node.next = current.next
                current.next = new_node
                self.linked_list._size += 1
                return
            current = current.next

    def add_after_recursive(self, el: object, n: object) -> None:
        def _add_after_recursive(node, el, n):
            if node is None:
                return
            if node.data == el:
                new_node = Node(n)
                new_node.next = node.next
                node.next = new_node
                self.linked_list._size += 1
                return
            _add_after_recursive(node.next, el, n)

        _add_after_recursive(self.linked_list.head, el, n)

    def get_at(self, index: int) -> object:
        return self.linked_list.get_at(index)

    def __iter__(self):
        return iter(self.linked_list)

class TestCollections(unittest.TestCase):

    def test_linked_list(self):
        ll = LinkedList()
        self.assertTrue(ll.is_empty())
        ll.add_first(1)
        ll.add_last(2)
        ll.add_at(3, 1)
        self.assertEqual(ll.size(), 3)
        self.assertEqual(ll.first(), 1)
        self.assertEqual(ll.last(), 2)
        self.assertEqual(ll.get_at(1), 3)
        self.assertTrue(ll.contains(2))
        self.assertFalse(ll.contains(4))
        ll.replace(2, 4)
        self.assertTrue(ll.contains(4))
        ll.remove_first()
        self.assertEqual(ll.first(), 3)
        ll.remove_last()
        self.assertEqual(ll.last(), 3)
        ll.remove_at(0)
        self.assertTrue(ll.is_empty())

    def test_double_linked_list(self):
        dll = DoubleLinkedList()
        self.assertTrue(dll.is_empty())
        dll.add_first(1)
        dll.add_last(2)
        dll.add_at(3, 1)
        self.assertEqual(dll.size(), 3)
        self.assertEqual(dll.first(), 1)
        self.assertEqual(dll.last(), 2)
        self.assertEqual(dll.get_at(1), 3)
        self.assertTrue(dll.contains(2))
        self.assertFalse(dll.contains(4))
        dll.replace(2, 4)
        self.assertTrue(dll.contains(4))
        dll.remove_first()
        self.assertEqual(dll.first(), 3)
        dll.remove_last()
        self.assertEqual(dll.last(), 3)
        dll.remove_at(0)
        self.assertTrue(dll.is_empty())

    def test_array_list(self):
        al = ArrayList(5)
        self.assertTrue(al.is_empty())
        al.add_first(1)
        al.add_last(2)
        al.add_at(3, 1)
        self.assertEqual(al.size(), 3)
        self.assertEqual(al.first(), 1)
        self.assertEqual(al.last(), 2)
        self.assertEqual(al.get_at(1), 3)
        self.assertTrue(al.contains(2))
        self.assertFalse(al.contains(4))
        al.replace(2, 4)
        self.assertTrue(al.contains(4))
        al.remove_first()
        self.assertEqual(al.first(), 3)
        al.remove_last()
        self.assertEqual(al.last(), 3)
        al.remove_at(0)
        self.assertTrue(al.is_empty())

    def test_array_list_stack(self):
        als = ArrayListStack()
        self.assertTrue(als.is_empty())
        als.push(1)
        als.push(2)
        als.push(3)
        self.assertEqual(als.size(), 3)
        self.assertEqual(als.top(), 3)
        self.assertEqual(als.pop(), 3)
        self.assertEqual(als.top(), 2)
        als.empty()
        self.assertTrue(als.is_empty())

    def test_linked_list_stack(self):
        lls = LinkedListStack()
        self.assertTrue(lls.is_empty())
        lls.push(1)
        lls.push(2)
        lls.push(3)
        self.assertEqual(lls.size(), 3)
        self.assertEqual(lls.top(), 3)
        self.assertEqual(lls.pop(), 3)
        self.assertEqual(lls.top(), 2)
        lls.empty()
        self.assertTrue(lls.is_empty())

    def test_linked_list_queue(self):
        llq = LinkedListQueue()
        self.assertTrue(llq.is_empty())
        llq.enqueue(1)
        llq.enqueue(2)
        llq.enqueue(3)
        self.assertEqual(llq.size(), 3)
        self.assertEqual(llq.front(), 1)
        self.assertEqual(llq.dequeue(), 1)
        self.assertEqual(llq.front(), 2)
        llq.empty()
        self.assertTrue(llq.is_empty())

    def test_circular_array_deque(self):
        cad = CircularArrayDeque(5)
        self.assertTrue(cad.is_empty())
        cad.enqueue(1)
        cad.enqueue(2)
        cad.enqueue(3)
        self.assertEqual(cad.size(), 3)
        self.assertEqual(cad.front(), 1)
        cad.left_enqueue(0)
        self.assertEqual(cad.front(), 0)
        self.assertEqual(cad.right_dequeue(), 3)
        self.assertEqual(cad.size(), 3)
        cad.dequeue()
        self.assertEqual(cad.front(), 1)
        cad.empty()
        self.assertTrue(cad.is_empty())

    def test_recursive_contains(self):
        lls = LinkedListStack()
        lls.push(1)
        lls.push(2)
        lls.push(3)
        self.assertTrue(recursive_contains(2, lls))
        self.assertFalse(recursive_contains(4, lls))

    def test_single_linked_list_add_before(self):
        sll = SingleLinkedList()
        sll.add_first(1)
        sll.add_last(3)
        sll.add_before(3, 2)  # Iterative
        self.assertEqual(sll.get_at(1), 2)
        sll.add_before_recursive(2, 1.5)  # Recursive
        self.assertEqual(sll.get_at(1), 1.5)

    def test_linked_list_stack_add_after(self):
        lls = LinkedListStack()
        lls.push(1)
        lls.push(3)
        lls.add_after(1, 2)  # Iterative
        self.assertEqual(lls.top(), 3)
        lls.add_after_recursive(3, 3.5)  # Recursive
        self.assertEqual(lls.get_at(1), 3.5)

    def test_single_linked_list_reorder(self):
        sll = SingleLinkedList()
        for i in range(1, 6):
            sll.add_last(i)
        sll.reorder()
        self.assertEqual(sll.get_at(0), 1)
        self.assertEqual(sll.get_at(1), 5)
        self.assertEqual(sll.get_at(2), 2)
        self.assertEqual(sll.get_at(3), 4)
        self.assertEqual(sll.get_at(4), 3)

if __name__ == '__main__':
    unittest.main()
