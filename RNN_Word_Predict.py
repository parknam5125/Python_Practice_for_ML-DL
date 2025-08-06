import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

chars = "abcdefghijklmnopqrstuvwxyz"
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

def predict_word(model, input_word):
    model.eval()
    hidden = model.init_hidden()
    one_hot = torch.from_numpy(word_to_onehot(input_word)).float()  # 🔧 여기 수정

    predicted = ""
    for i in range(1, len(one_hot) - 1):
        input_letter = one_hot[i]
        output, hidden = model(input_letter, hidden)

        predicted_index = torch.argmax(output).item()
        predicted_char = char_list[predicted_index]
        predicted += predicted_char

    return predicted

def main():
    n_hidden = 128
    lr = 0.001
    epochs = 1000

    GT_list = [
        "basic", "beach", "below", "black", "brown", "carry", "cream", "drink", "error", "event",
    ]

    rnn = myRNN(n_letters, n_hidden, n_letters)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for string in GT_list:
            one_hot = torch.from_numpy(word_to_onehot(string)).float()
            rnn.zero_grad()
            hidden = rnn.init_hidden()

            loss = 0
            for i in range(len(string) + 1):
                input_letter = one_hot[i]
                target_letter = torch.tensor([torch.argmax(one_hot[i + 1])])
                output, hidden = rnn(input_letter, hidden)
                loss += loss_func(output, target_letter)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"\n[Epoch {epoch}] Total Loss: {total_loss:.4f}")
            for string in GT_list[:10]:
                predicted = predict_word(rnn, string)
                print(f"GT: {string}  OUTPUT: {predicted}")

if __name__ == "__main__":
    main()
