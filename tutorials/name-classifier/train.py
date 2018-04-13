import random
import time
import math

import torch
import torch.nn as nn
from torch.autograd import Variable 

import matplotlib.pyplot as pyplot
import matplotlib.ticker as ticker

from load import all_categories, all_letters, n_categories, n_letters, category_lines

def letter_to_index(letter):
    return all_letters.find(letter)

def letter_to_tensor(letter):
    index = letter_to_index(letter)

    tensor = torch.zeros(1, n_letters)
    tensor[0][index] = 1

    return tensor

def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)

    for i, letter in enumerate(line):
        tensor[i][0][letter_to_index(letter)] = 1

    return tensor

def category_from_output(output):
    top_n, top_i = output.data.topk(1)
    category_i = top_i[0][0]

    return all_categories[category_i], category_i

def random_choice(items):
    return random.choice(items)

def random_training_sample():
    category = random_choice(all_categories)
    line = random_choice(category_lines[category])

    category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
    line_tensor = Variable(line_to_tensor(line))

    return category, line, category_tensor, line_tensor

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.h2o(combined)
        output = self.softmax(output)

        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

hidden_size = 128
learning_rate = 0.05

criterion = nn.NLLLoss()
rnn = RNN(n_letters, hidden_size, n_categories)

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()
    rnn.zero_grad()

    for i in range(0, line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.data[0]

n_iters = 100000
print_every = 5000
plot_every = 1000

current_loss = 0
all_losses = []

def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for i in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = random_training_sample()
    output, loss = train(category_tensor, line_tensor)

    current_loss += loss
    if i % print_every == 0:
        guess, guess_i = category_from_output(output)
        correct = '✓' if guess == category else f'✗ ({category})'
        print(f'{i} {i / n_iters * 100}% ({time_since(start)}) {loss} {line} / {guess} {correct}')

    if i % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

pyplot.figure()
pyplot.plot(all_losses)
pyplot.show()
