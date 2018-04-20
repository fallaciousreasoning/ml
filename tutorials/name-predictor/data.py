from __future__ import unicode_literals, division

import glob

import unicodedata
import string

import torch

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1 # Plus EOS marker

def find_files(path): return glob.glob(path)

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def read_lines(filename):
    with open(filename) as f:
        return [unicode_to_ascii(line.strip()) for line in f.readlines()]

category_lines = {}
all_categories = []
for filename in find_files('data/names/*.txt'):
    category = filename.split('/')[-1].split('.')[0]
    all_categories.append(category)
    lines = read_lines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

def category_tensor(category):
    index = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][index] = 1
    return tensor

def line_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for i in range(len(line)):
        letter = line[i]
        tensor[i][0][all_letters.find(letter)] = 1

    return tensor

def target_tensor(line):
    letter_indexes = [all_letters.find(letter) for letter in line[1:]]
    letter_indexes.append(n_letters - 1) # EOL
    return torch.LongTensor(letter_indexes)