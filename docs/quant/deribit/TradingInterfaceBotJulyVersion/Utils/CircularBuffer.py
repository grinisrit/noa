from typing import List, Generic, TypeVar

T = TypeVar("T")


class CircularBuffer(Generic[T]):
    _data: List[T]

    def __init__(self, size):
        """initialization"""
        self.index = 0
        self.size = size
        self._data: List[T] = []

    def record(self, value):
        """append an element"""
        if len(self._data) == self.size:
            self._data[self.index] = value
        else:
            self._data.append(value)
        self.index = (self.index + 1) % self.size

    def __getitem__(self, key) -> T:
        """Get element by index, relative to the current index"""
        if len(self._data) == 0:
            return None

        if len(self._data) == self.size:
            return self._data[(key + self.index) % self.size]
        else:
            return self._data[key]

    def __repr__(self):
        """return string representation"""
        return self._data.__repr__() + ' (' + str(len(self._data))+' items)'

    def get_all(self) -> List[T]:
        """return a list of all the elements"""
        return self._data
