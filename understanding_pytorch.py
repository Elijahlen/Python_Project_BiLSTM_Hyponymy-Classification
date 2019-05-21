'''
to create a simple cnn there are two ways.
1. create class(torch.nn.module) inital super def forward
2. quick way, sequential(linear, RUel, linear)
    
'''

'''
two ways to save the network
1.torch.save(net1, 'net.pkl')
2.torch.save(net1.sate_dict(), 'net_params.plk')
'''

'''
two ways to load network
1. load directily net2 = torch.load('net.pkl')
2. create the structure first and load the paramaters.
net3 = torch.nn.Sequential(
    torch.nn.Linear(1,10),
    torch.nn.ReLU(),
    torch.nn.Linear(10,1)
)
net3.load_state_dict(torch.load('net_params.pkl'))

'''
'''
regression example
'''
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as Data

x1 = torch.linspace(-1,1,100)
x = torch.unsqueeze(x1, dim = 1)
# print(x.size())
y = x.pow(2) + 0.2*torch.rand(x.size())

# plt.scatter(x.data.numpy(),y.data.numpy())
# plt.show()

class Net(torch.nn.Module):
    def __init__(self, n_input, n_hidden_put, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_input, n_hidden_put)
        self.predict = torch.nn.Linear(n_hidden_put, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x
net = Net(1,10,1)
print(net)

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
loss_func = torch.nn.MSELoss()

for t in range(100):
    prediction = net(x)

    loss = loss_func(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

