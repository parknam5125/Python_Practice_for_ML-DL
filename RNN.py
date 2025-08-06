import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

chars = "abcdefghijklmnopqrstuvwxyz .,;?01"
char_list = [i for i in chars]
n_letters = len(char_list)

def word_to_onehot(string):
    start = np.zeros(shape=len(char_list), dtype=int)
    end = np.zeros(shape=len(char_list), dtype=int)
    start[-2] = 1
    end[-1] = 1
    for i in string:
        idx = char_list.index(i)
        zero = np.zeros(shape=n_letters, dtype=int)
        zero[idx] = 1
        start = np.vstack([start, zero])
    output = np.vstack([start, end])
    return output

def onehot_to_word(onehot_1):
    onehot = torch.Tensor.numpy(onehot_1)
    return char_list[onehot.argmax()]

class myRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(myRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.U = nn.Linear(input_size, hidden_size)
        self.W = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, output_size)
        self.activation_func = nn.Tanh()
        self.softmax = nn.LogSoftmax(dim=0)

    def forward(self, input, hidden):
        hidden = self.activation_func(self.U(input) + self.W(hidden))
        output = self.V(hidden)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

def main():
    n_hidden = 128
    lr = 0.001
    epochs = 1000
    string = "i want to go on a trip these days. how about you?"

    rnn = myRNN(n_letters, n_hidden, n_letters)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)

    one_hot = torch.from_numpy(word_to_onehot(string)).type_as(torch.FloatTensor())

    for i in range(epochs):
        optimizer.zero_grad()
        total_loss = 0
        hidden = rnn.init_hidden()

        input_ = one_hot[0:1, :]
        for j in range(one_hot.size()[0] - 1):
            target = one_hot[j+1]
            target_single = torch.from_numpy(np.array(target.numpy().argmax())).type_as(torch.LongTensor()).view(-1)

            output, hidden = rnn.forward(input_, hidden)
            loss = loss_func(output, target_single)
            total_loss += loss
            input_ = output

        total_loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print('epoch%d' % i)
            print(total_loss)

            start = torch.zeros(1, len(char_list))
            start[:, -2] = 1
            with torch.no_grad():
                hidden = rnn.init_hidden()
                input_ = start
                output_string = ""
                for _ in range(len(string)):
                    output, hidden = rnn.forward(input_, hidden)
                    output_string += onehot_to_word(F.softmax(output.data))
                    input_ = output
                print(output_string)

if __name__ == '__main__':
    main()
