from typing import Protocol

class ImageDataObject(Protocol):
    def batch_and_shuffle(self, batch_size: int, buffer_size: int):
        ...
