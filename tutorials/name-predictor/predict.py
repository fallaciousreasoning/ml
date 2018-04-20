import torch
from torch.autograd import Variable

from data import category_tensor, line_tensor, n_letters, all_letters, all_categories

max_length = 20
rnn = torch.load('char-rnn-predict.pt')

def sample(category, start_letter):
    cat_ten = Variable(category_tensor(category))
    line = Variable(line_tensor(start_letter))

    hidden = rnn.init_hidden()

    output_name = start_letter

    for i in range(max_length):
        output, hidden = rnn(cat_ten, line[0], hidden)
        topv, topi = output.topk(1)
        topi = topi[0][0].data[0]

        if topi == n_letters - 1: # if it's an EOL
            break
        
        letter = all_letters[topi]
        output_name += letter
        line = Variable(line_tensor(letter))

    return output_name

def generate_samples(category, start_letters=None):
    if start_letters is None:
        start_letters = category

    for letter in start_letters:
        print(sample(category, letter.upper()))

if __name__ == '__main__':
    for category in all_categories:
        print(category)
        generate_samples(category)
        print()