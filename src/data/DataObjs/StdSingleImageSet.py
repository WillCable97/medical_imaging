

class StdSingleImageSet:
    def __init__(self, input_func, **kwargs):
        image_data = input_func(kwargs) #May need conversion, this is an np array
        
    def batch_and_shuffle(self, batch_size: int, buffer_size: int):
        pass