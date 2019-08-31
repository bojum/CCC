from __future__ import print_function
import torch, csv

### Sample code from pytorch tutorial
###
###

### read in target and attr input files
target_file = 'target_in.csv'
attr_file = 'attr_in.csv'

with open(target_file, 'r') as f:
    reader = csv.reader(f)
    target_input_list = list(reader)

with open(attr_file, 'r') as f:
    reader = csv.reader(f)
    attr_input_list = list(reader)

## unnest list of target input
target_input = [t for sublist in target_input_list for t in sublist]

## convert str attr to float attr
attr_input = [[float(x) for x in t] for t in attr_input_list]


#dtype = torch.float
#device = torch.device("cpu")

# N = number of instances per 1 feed
# D_in = number of attributes
# D_out = number of output attributes e.g. num of classes
N, D_in, H, D_out = 2, 2675, 100, 4

##random tensors
#x = torch.randn(N, D_in) #, device = device) #, dtype = dtype, requires_grad = True)
#y = torch.randn(N, D_out) #, device = device) #, dtype = dtype, requires_grad = True)

model = torch.nn.Sequential(
torch.nn.Linear(D_in, H),
torch.nn.ReLU(),
torch.nn.Linear(H, D_out),
)

loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
for t in range(500):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    model.zero_grad()
    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
#x = torch.tensor([[1, 3], [2,3], [4, 5]])
#y = torch.tensor([2, 5])

#w1 = torch.tensor([[-1, 3, 2], [-3, 4, 5]])
#w2 = torch.tensor([-1,-1])


##tensors for weights
#w1 = torch.randn(D_in, H, device = device, dtype = dtype)

#x = torch.rand(5,3)
