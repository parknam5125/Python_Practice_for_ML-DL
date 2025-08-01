import numpy as np
import torch

x=torch.FloatTensor([[80,220,6300],
                     [75, 167, 4500],
                     [86,210,7500],
                     [110,330,9000],
                     [95,280,8700],
                     [67,190,6800],
                     [79,210,5000],
                     [98,250,7200]])
y=torch.LongTensor([2,3,1,0,0,3,2,1])

print(x.shape)
print(y.shape)

W = torch.zeros((3,4),requires_grad=True)
b = torch.zeros(1,requires_grad=True)

optimizer = torch.optim.SGD([W,b], lr = 0.00001)

for e in range(1000):
    optimizer.zero_grad()
    z = x.matmul(W)+b
    cost = torch.nn.functional.cross_entropy(z,y)
    cost.backward()
    optimizer.step()
    if(e%100==0):
        print('epoch:%d cost:%f'%(e,cost))

test = torch.FloatTensor([[85,250,7000]])

z = test.matmul(W) + b
predict = torch.softmax(z, dim=1)
pred_np = predict.detach().numpy()
rounded_pred = np.round(pred_np, 4)

print("prob:", rounded_pred)

predicted_class = torch.argmax(predict, dim=1)
print("expect:", predicted_class.item())
