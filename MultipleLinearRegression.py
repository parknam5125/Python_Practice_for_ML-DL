import torch
import numpy as np

x = torch.FloatTensor([[3.8, 700, 80, 50],
                       [3.2, 650, 90, 30],
                       [3.7, 820, 70, 40],
                       [4.2, 830, 50, 70],
                       [2.6, 550, 90, 60],
                       [3.4, 910, 30, 40],
                       [4.1, 990, 70, 20],
                       [3.3, 870, 60, 60]])

y = torch.FloatTensor([[85],[80],[78],[87],[85],[70],[81],[88]])

W = torch.zeros((4,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


optimizer = torch.optim.SGD([W,b], lr=0.000001)

epochs=100
for epoch in range(epochs):
    hypothesis = x.mm(W) + b
    cost = torch.mean((hypothesis - y) ** 2)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    print('epoch:%d cost:%f'%(epoch, cost))

test=torch.FloatTensor([[3.3, 700, 77, 84]])
print(W)
print(b)
predict=test.mm(W)+b
pred_val=predict.squeeze().detach().numpy()
print("Total score is estimated as %d"%(pred_val))