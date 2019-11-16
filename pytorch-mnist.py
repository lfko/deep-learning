"""
    toy example using the MNIST data set with PyTorch
    https://nextjournal.com/gkoehler/pytorch-mnist
    https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627
"""
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

batch_size_train = 32
batch_size_test = 500
num_workers = 1

class Net(nn.Module):
    def __init__(self, num_classes = 10):
        super(Net, self).__init__()

        # Sequential for grouping blocks/layers
        self.features = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(), 
            nn.Linear(128, 64),
            nn.ReLU()
        )
        # A single model can be used to simulate having a large number of different network architectures by randomly dropping out nodes during training. 
        # This is called dropout and offers a very computationally cheap and remarkably effective regularization method to reduce overfitting 
        # and improve generalization error in deep neural networks of all kinds.
        #self.maxpool = nn.MaxPool2d(2, 2)
        self.classifier = nn.Sequential(
            nn.Linear(64, 10), nn.LogSoftmax(dim = 1)
        )
    
    def forward(self, x):
        '''
            defines the way how the output is computed with the given layers and functions
        '''
        # pools do some kind of feature selection, choosing the most dominant features from the image
        # or combining different ones
        #x = F.relu(F.max_pool2d(self.conv1(x), 2)) # IN -> RELU -> H1
        #x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)) # H1 -> RELU -> H2
        #x = x.view(-1, 320)
        #x = self.relu(self.conv1(x))
        #x = self.pool(x) 
        #x = self.relu(self.conv2(x))
        #x = self.pool(x) 
        #x = x.view(-1, 28 * 28 * 10)

        #x = self.fc2(x) # final output layer

        # Sequential
        x = self.features(x)
        #x = self.maxpool(x)
        x = x.view(-1, 64)
        x = self.classifier(x)

        return x

# load the MNIST data set (automatically split into train and test)
mnist_train = torchvision.datasets.MNIST('./data', train = True, download = True)
print(len(mnist_train))
mnist_train_mean = mnist_train.train_data.float().mean() / 255 # gray images
mnist_train_std = mnist_train.train_data.float().std() / 255

print('Train mean:', mnist_train_mean)
print('Train std:', mnist_train_std)

# Transformer for transforming the data to Tensors and normalizing it
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mnist_train_mean, ), (mnist_train_std, ))])
mnist_train = torchvision.datasets.MNIST('./data', train = True, download = True, transform = transform)
mnist_test = torchvision.datasets.MNIST('./data', train = False, download = True, transform = transform)

# use num_workers > 1 to use subprocesses to asynchronously load data or using pinned RAM (via pin_memory) to speed up RAM to GPU transfers
# Training loaders for batch-learning (32 at once)
trainloader = torch.utils.data.DataLoader(mnist_train, batch_size = batch_size_train, shuffle = True, num_workers = num_workers)
# Same for test set
testloader = torch.utils.data.DataLoader(mnist_test, batch_size = batch_size_test, shuffle = True, num_workers = num_workers)

# print out shapes
train_set_shape = trainloader.dataset.train_data.shape
test_set_shape = testloader.dataset.test_data.shape

print('Shape Train: ', train_set_shape, 'Shape Test:', test_set_shape)

# initialise a new Network
net = Net()
optimizer = optim.SGD(net.parameters(), lr = .01, momentum = .5)
criterion = nn.CrossEntropyLoss()

# actual training
net.train()
for epoch in range(10):
    for batch_idx, data in enumerate(trainloader):
        # get inputs
        inputs, labels = data
        inputs = inputs.view(-1, 28 * 28 * 1)

        optimizer.zero_grad()

        # Forward + Backward + Optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward() # computes the gradients
        optimizer.step() # adapt the weights

# prediction on unseen data
correct, total = 0, 0
#preds = []
net.eval() # net is set to test/eval mode

for i, data in enumerate(testloader):
    inputs, labels = data
    inputs = inputs.view(-1, 28 * 28 * 1)
    outputs = net(inputs) # calculates scores for each class
    _, outputs = torch.max(outputs.data, 1) # get class with highest score
    #preds.append(outputs)
    total += labels.size(0)
    correct += (outputs == labels).sum().item()

print('Test accuracy: %d %%' %(100 * correct / total))