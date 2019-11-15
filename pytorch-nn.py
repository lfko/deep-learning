"""
    https://campus.datacamp.com/courses/deep-learning-with-pytorch
"""
import torch
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

    simple_computational_graph()