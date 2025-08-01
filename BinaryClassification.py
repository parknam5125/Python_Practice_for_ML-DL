import numpy as np
import torch

x=torch.FloatTensor([[80,220],[75,167],[86,210],[110,330],[95,280],[67,190],[79,210],[98,250]])
y=torch.FloatTensor([[1],[0],[1],[1],[1],[0],[0],[1]])

print(x.shape)
print(y.shape)

W = torch.zeros((2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = torch.optim.SGD([W,b],lr=0.00001)

for e in range(100):
    optimizer.zero_grad()
    hypothesis = torch.sigmoid(x.matmul(W) + b)
    cost = torch.nn.functional.binary_cross_entropy(hypothesis,y)
    cost.backward()
    optimizer.step()

    print('epoch:%d cost:%f'%(e, cost))

test = torch.FloatTensor([[90,200]])

predict = torch.sigmoid(test.matmul(W)+b)

print('probability:', predict.item())