import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

chars = "abcdefghijklmnopqrstuvwxyz"
char_list = [i for i in chars]
n_letters = len(char_list)

n_layers = 1

five_words = [
    'basic', 'beach', 'below', 'black', 'brown', 'carry', 'cream', 'drink', 'error', 'event',
    'exist', 'first', 'funny', 'guess', 'human', 'image', 'large', 'magic', 'mouse', 'night',
    'noise', 'ocean', 'often', 'order', 'peace', 'phone', 'print', 'quiet', 'reach', 'rough',
    'round', 'scene', 'score', 'sense', 'skill', 'sleep', 'small', 'storm', 'table', 'think',
    'touch', 'twice', 'until', 'upset', 'voice', 'waste', 'watch', 'white', 'woman', 'young'
]
n_five_words = len(five_words)

sequence_length = 4

def word_to_onehot(string):
    one_hot = np.zeros((len(string), n_letters), dtype=np.float32)
    for i,ch in enumerate(string):
        idx = char_list.index(ch)
        one_hot[i][idx] = 1
    return one_hot

def onehot_to_word(onehot_1):
    onehot = torch.Tensor.numpy(onehot_1)
    return char_list[onehot.argmax()]

class myLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer):
        super(myLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.LSTM = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layer, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x, hidden):
        out, hidden = self.LSTM(x, hidden)
        out = self.fc(out)
        return out, hidden
    
    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.num_layer, batch_size, self.hidden_size),
                torch.zeros(self.num_layer, batch_size, self.hidden_size))

def main():
    n_hidden = 26
    lr = 0.001
    epochs = 900
    
    model = myLSTM(n_letters, n_hidden, n_layers)
    
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.1)
    
    for i in range(epochs):
        total_loss = 0
        for j in range(n_five_words):
            string = five_words[j]
            one_hot = torch.from_numpy(word_to_onehot(string)).float()
            input = one_hot[0:-1].unsqueeze(0)
            target = torch.tensor(np.argmax(one_hot[1:], axis=1))

            hidden = model.init_hidden(batch_size=1)
            model.zero_grad()
            output, hidden = model(input, hidden)
            output = output.squeeze(0)

            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if i % 10 == 0:
            print('epoch%d' % i)
            print(loss)
        
        scheduler.step()
    
    torch.save(model.state_dict(), 'trained.pth')
    model.load_state_dict(torch.load('trained.pth'))
    
    with torch.no_grad():
        total = 0
        correct_last = 0
        total_chars = 0
        correct_chars = 0

        for i in range(n_five_words):
            string = five_words[i]
            one_hot = torch.from_numpy(word_to_onehot(string)).float()
            input = one_hot[0:-1].unsqueeze(0)
            target = string[1:]

            hidden = model.init_hidden(batch_size=1)
            output, hidden = model(input, hidden)
            output = output.squeeze(0)

            output_string = string[0]

            for j in range(output.size(0)):
                output_char = onehot_to_word(output[j].data)
                output_string += output_char
                total_chars += 1
                if output_char == target[j]:
                    correct_chars += 1

            total += 1
            if output_string[-1] == string[-1]:
                correct_last += 1

            print(f"{i + 1:2d} GT:{string} OUT:{output_string}")

        print(f"Final text accuracy: {correct_last}/{total} ({correct_last / total:.4f})")
        print(f"Whole text accuracy: {correct_chars}/{total_chars} ({correct_chars / total_chars:.4f})")

if __name__ == '__main__':
    main()