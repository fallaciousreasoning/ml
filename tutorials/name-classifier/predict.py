import sys

import torch
from torch.autograd import Variable

from load import all_categories
from shared import line_to_tensor, category_from_output

rnn = torch.load('char-rnn-classification.pt')

def evaluate(line_tensor):
    hidden = rnn.initHidden()
    
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    
    return output

def predict(line, n_predictions=3):
    print(f'\n> {line}')
    output = evaluate(Variable(line_to_tensor(line)))

    # Get top N categories
    topv, topi = output.data.topk(n_predictions, 1, True)
    predictions = []

    for i in range(n_predictions):
        value = topv[0][i]
        category_index = topi[0][i]
        print('(%.2f) %s' % (value, all_categories[category_index]))
        predictions.append([value, all_categories[category_index]])

    return predictions

if __name__ == '__main__':
    if len(sys.argv) > 1:
        predict(sys.argv[1])
    else:
        predict('Jay')
        predict('Cheng')
        predict('Dostoyevsky')