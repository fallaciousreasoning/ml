import random

import time

import math

import torch
import torch.nn as nn
from torch.autograd import Variable

from rnn import RNN

from data import all_categories, all_letters, category_lines, category_tensor, line_tensor, target_tensor, n_letters

def random_line(): 
    category = random.choice(all_categories)
    line = random.choice(category_lines[category])

    return category, line

def random_sample():
    category, line = random_line()
    c = Variable(category_tensor(category))
    l = Variable(line_tensor(line))
    t = Variable(target_tensor(line))

    return c, l, t

def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

hidden_size = 128
learning_rate = 0.0005

rnn = RNN(n_letters, hidden_size, n_letters)

def train(category_tensor, line_tensor, target_tensor):
    hidden = rnn.init_hidden()

    rnn.zero_grad()

    loss = 0

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(category_tensor, line_tensor[i], hidden)
        loss += criterion(output, target_tensor[i])

    loss.backward()

    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.data[0] / line_tensor.size()[0]

n_iterations = 100000
print_every = n_iterations/100
plot_every = 1000

criterion = nn.NLLLoss()

total_loss = 0
all_losses = []

start = time.time()

for i in range(1, n_iterations + 1):
    output, loss = train(*random_sample())
    total_loss += loss

    if i % print_every == 0:
        print('%s (%d %d%%) %.4f' % (time_since(start), i, i / n_iterations * 100, loss))

    if i % plot_every == 0:
        all_losses.append(total_loss / plot_every)
        total_loss = 0

torch.save(rnn, 'char-rnn-predict.pt')

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)