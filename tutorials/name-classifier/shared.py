import random

import torch
from torch.autograd import Variable

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