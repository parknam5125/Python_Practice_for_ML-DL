import torch
import numpy as np

x = torch.FloatTensor([[78],[83],[56],[67],[85],[44],[32],[90]])
y = torch.FloatTensor([[66],[73],[76],[65],[81],[54],[29],[85]])

W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)


optimizer = torch.optim.SGD([W,b], lr=0.0001)

epochs=100
for epoch in range(epochs):
    hypothesis = W * x + b
    cost = torch.mean((hypothesis - y) ** 2)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    print('epoch:%d cost:%f'%(epoch, cost))

print(W)
print(b)
predict=W*71+b
print('My final score is estimated as %d'%(predict))