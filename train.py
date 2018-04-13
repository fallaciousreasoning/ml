import torch

class Data:
    def __init__(self, filename):
        self.filename = filename
    
    def load(self):
        data = None

        with open(self.filename) as f:
            data = f.read()

        self.data = data
        self.chars = list(set(self.data))
        self.data_size = len(self.data)
        self.vocab_size = len(self.chars)

        self.char_to_id = { char:id for id,char in enumerate(self.chars) }
        self.id_to_char = { id:char for id,char in enumerate(self.chars) }


def load_data(filename):
    data = Data(filename)

    data.load()

    print(f'Data has {data.data_size} characters, of which {data.vocab_size} are unique')
    return data

load_data('data/shakespeare.txt')


hidden_size = 100
seq_length = 25
learning_rate = 1e-1