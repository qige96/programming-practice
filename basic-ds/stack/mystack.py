"""
Python stack implementation, using built-in list.
"""

class Stack:
    """Stack implementation using built-in list."""
    def __init__(self):
        self._list = []

    def __len__(self):
        return len(self._list)

    def is_empty(self):
        return len(self._list) == 0

    def push(self, item):
        self._list.append(item)

    def pop(self):
        if self.is_empty():
            raise IndexError("Pop from empty stack!")
        else:
            item = self._list[-1]
            del self._list[-1]
            return item
