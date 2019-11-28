"""
Python queue implementation, using built-in list. 
"""

class Queue:
    """Queue implementation using built-in list"""
    def __init__(self):
        self._list = []

    def __len__(self):
        return len(self._list)

    def is_empty(self):
        return len(self._list) == 0

    def enqueue(self, item):
        self._list.append(item)

    def dequeue(self):
        if self.is_empty():
            raise IndexError("Dequeue from empty queue!")
        else:
            item = self._list[0]
            del self._list[0]
            return item

