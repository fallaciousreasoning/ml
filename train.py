import torch
from torch.autograd import Variable

dtype = torch.FloatTensor

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

        self.char_to_id = {char: id for id, char in enumerate(self.chars)}
        self.id_to_char = {id: char for id, char in enumerate(self.chars)}


def load_data(filename):
    data = Data(filename)

    data.load()

    print(f'Data has {data.data_size} characters, of which {data.vocab_size} are unique')
    return data


data_info = load_data('data/shakespeare.txt')
data = data_info.data
data_size = data_info.data_size
chars = data_info.chars
vocab_size = data_info.vocab_size
char_to_id = data_info.char_to_id
id_to_char = data_info.id_to_char

hidden_size = 100
seq_length = 200
learning_rate = 1e-1

input_weights = Variable(torch.randn(hidden_size, vocab_size).type(dtype) * 0.01, require_grad=True)
hidden_weights = Variable(torch.randn(hidden_size, hidden_size).type(dtype) * 0.01, require_grad=True)
output_weights = Variable(torch.randn(hidden_size, hidden_size).type(dtype) * 0.01, require_grad=True)

hidden_biases = Variable(torch.zeros(hidden_size, 1), require_grad=False)
output_biases = Variable(torch.zeros(vocab_size, 1), require_grad=False)


def lossFun(inputs, targets, hprev):
    """
    inputs,targets are both list of integers.
    hprev is Hx1 array of initial hidden state
    returns the loss, gradients on model parameters, and last hidden state
    """
    input_at_t, hidden_at_t, output_at_t, ps = {}, {}, {}, {}
    loss = 0

    # for each character in the input
    for t in range(len(inputs)):
        input_at_t[t] = torch.zeros(vocab_size, 1)  # encode in 1-of-k representation
        input_at_t[t][inputs[t]] = 1 # set the current character as true

        x = input_at_t[t]

        y_pred = x.mm(input_weights).clamp(min=0).mm(w2)
        
        hs[t] = (input_weights.mm(xs[t]) + hidden_weights.mm(hs[t - 1]) + hidden_biases).tanh()

        # unnormalized log probabilities for next chars
        ys[t] = hidden_weights.mm(hs[t]) + output_biases

        # probabilities for next chars
        ps[t] = (ys[t] / ys[t].exp().sum()).exp()
        
        loss += -np.log(ps[t][targets[t], 0])  # softmax (cross-entropy loss)
    # backward pass: compute gradients going backwards
    


def sample(h, seed_ix, n):
    """ 
    sample a sequence of integers from the model 
    h is memory state, seed_ix is seed letter for first time step
    """
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []
    for t in range(n):
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        y = np.dot(Why, h) + by
        p = np.exp(y) / np.sum(np.exp(y))
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        ixes.append(ix)
    return ixes


n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by)  # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length  # loss at iteration 0
while True:
    # prepare inputs (we're sweeping from left to right in steps seq_length long)
    if p+seq_length+1 >= data_size or n == 0:
        hprev = np.zeros((hidden_size, 1))  # reset RNN memory
        p = 0  # go from start of data
    inputs = [char_to_id[ch] for ch in data[p:p+seq_length]]
    targets = [char_to_id[ch] for ch in data[p+1:p+seq_length+1]]

    # sample from the model now and then
    if n % 100 == 0:
        sample_ix = sample(hprev, inputs[0], 200)
        txt = ''.join(id_to_char[ix] for ix in sample_ix)
        print('----\n %s \n----' % (txt, ))

    # forward seq_length characters through the net and fetch gradient
    loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    if n % 100 == 0:
        print('iter %d, loss: %f' % (n, smooth_loss))  # print progress

    # perform parameter update with Adagrad
    for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                  [dWxh, dWhh, dWhy, dbh, dby],
                                  [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / \
            np.sqrt(mem + 1e-8)  # adagrad update

    p += seq_length  # move data pointer
    n += 1  # iteration counter
