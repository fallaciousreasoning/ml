
import time
import math

import torch
import torch.nn as nn
from torch.autograd import Variable 

import matplotlib.pyplot as pyplot
import matplotlib.ticker as ticker

from load import all_categories, all_letters, n_categories, n_letters, category_lines
from shared import letter_to_index, letter_to_tensor, line_to_tensor, category_from_output, random_training_sample
from model import RNN

hidden_size = 256
learning_rate = 0.01

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

criterion = nn.NLLLoss()
rnn = RNN(n_letters, hidden_size, n_categories)

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

torch.save(rnn, 'char-rnn-classification.pt')

pyplot.figure()
pyplot.plot(all_losses)
pyplot.show()