"""
    https://campus.datacamp.com/courses/deep-learning-with-pytorch
"""
import torch
import torch.nn as nn
import numpy

def simple_computational_graph():
    '''
        Builds a simple computational graph, consisting of three large random tensors
    '''
    X = torch.rand((1000, 1000))
    Y = X.clone()
    Z = X.clone()

    print(X.shape, Y.shape, Z.shape)

    # first "layer"
    q = torch.matmul(X, Y)

    # second "layer"
    f = q * Z
    print(torch.mean(f))

def simple_backward_propagation():
    '''
        Backpropagation in use (very artifical)
    '''
    #X = torch.tensor(10., requires_grad = True) # auto-derivatives
    X = torch.rand((1000, 1000), requires_grad = True)
    Y = X.clone()
    Z = X.clone()
    #Y = torch.tensor(5., requires_grad = True)
    #Z = torch.tensor(-3., requires_grad = True)

    q = X + Y
    f = q * Z

    mean_f = torch.mean(f)
    mean_f.backward() # compute the derivatives

    print('Backpropagated weights/gradients:')
    print(X.grad)
    print(Y.grad)
    print(Z.grad)

def network_with_relu():
    '''
        Using ReLu as activation function
    '''
    relu = nn.ReLU()
    input_layer = torch.rand(4)

    weight_1 = torch.rand(4, 6) # IN <> H1
    weight_2 = torch.rand(6, 2) # H1 <> OUT

    h1_layer = torch.matmul(input_layer, weight_1)
    h1_layer_relued = relu(h1_layer)
    output_layer = torch.matmul(h1_layer_relued, weight_2)
    
    print(output_layer)


def simple_neural_network():
    '''
        1 hidden layer network, the hard way        
    '''
    input_layer = torch.rand(10) 
    w1 = torch.rand(10, 20) # weights between input and first hidden layer
    w2 = torch.rand(20, 20) # 
    w3 = torch.rand(20, 4) # weights between last hidden and output layer
    h1 = torch.matmul(input_layer, w1)
    h2 = torch.matmul(h1, w2)
    output_layer = torch.matmul(h2, w3)

    print(output_layer)

    # now the PyTorch way
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()

            # 2 linear layers
            self.fc1 = nn.Linear(784, 200)
            self.fc2 = nn.Linear(200, 10)

        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x)

            return x

    input_layer = torch.rand(784)
    net = Net()
    res = net(input_layer)
    print(res.shape)
    print(res)

if __name__ == "__main__":
    first_tensor = torch.rand((3, 3)) # 3x3 Tensor with random numbers
    ts_shape = first_tensor.shape
    print(ts_shape)

    ts_ones = torch.ones((3, 3)) # Tensor/Matrix of ones
    ts_ident = torch.eye(3) # np.idenity(3), idenity matrix

    # do matrix multiplication
    ts_res = torch.matmul(ts_ones, ts_ident) # np.dot()
    
    # ts_res = ts_ones * ts_ident element-wise multiplication

    # torch.eq() element-wise
    # torch.all() over all

    if (torch.all(torch.eq(ts_res, ts_ones)) == True):
        print("Yep, they are equal!")

    #simple_computational_graph()
    #simple_backward_propagation()
    #simple_neural_network()
    network_with_relu()