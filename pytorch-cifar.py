"""
    @author lfko
    @date 11/17/19

    @NB
    PyTorch CNN, CIFAR data set (https://blog.paperspace.com/pytorch-101-building-neural-networks/, http://pjreddie.com/media/files/cifar.tgz)
    https://www.edureka.co/blog/convolutional-neural-network/
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    https://www.stefanfiott.com/machine-learning/cifar-10-classifier-using-cnn-in-pytorch/
"""

# necessary imports
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.utils.data
import torch.optim as optim

import os
import random
import matplotlib.pyplot as plt

class MyCNN(nn.Module):
    def __init__(self, num_classes = 10):
        super(MyCNN, self).__init__()

        # input layer
        self.input = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 5, stride = 1, padding = 0),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels = 6, out_channels = 16, stride = 1, padding = 0, kernel_size = 5),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120), nn.ReLU(inplace = True),
            nn.Linear(120, 84), nn.ReLU(inplace = True)
        )
        self.out = nn.Sequential(
            nn.Linear(84, 10)
        )

    def forward(self, x):

        x = self.input(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.fc(x)

        return self.out(x)

class MyCIFARDataset(torch.utils.data.Dataset):
    """
        Basically a re-implementation of the already existing CIFAR dataset loader built-in in PyTorch
        Allows for lazy data loading
    """
    def __init__(self, data_dir = './data', data_size = 0, transforms = None):
        """
            data_dir - directory containing the dataset
            data_size
            transforms - Transformations which should be applied
        """
        files = os.listdir(data_dir) # read all files in the directory
        files = [os.path.join(data_dir, x) for x in files] # concats path + filename

        if len(files) == 0:
            raise FileNotFoundError('Supplied folder could not be loaded!')

        self.data_size = len(files)
        self.files = random.sample(files, self.data_size)
        self.transforms = transforms

    def __len__(self):
        """ Returns the length of the dataset """ 
        return self.data_size

    def __getitem__(self, idx):
        """ Returns element for a specific id """
        image_uri = self.files[idx] # retrieve image path
        image = plt.imread(image_uri)
        label_string = image_uri[image_uri.rfind('_') + 1: len(image_uri) - 4]
        label = readLabels()[label_string]

        if self.transforms:
            image = self.transforms(image) # apply some transformation before returning

        return image, label

def runCNN(model = None, mode = 'train', criterion = None, optimizer = None, 
            dataloader = None, epochs = 10):
    """

    """
    if model == None:
        assert('Please provide a valid model')

    print('epochs: ', epochs)
    print('mode:', mode)
    print('CUDA available? ', torch.cuda.is_available())

    if mode == 'train':
        model.train()

        for epoch in range(epochs):
            print('running epoch #', epoch)
            for batch_idx, data in enumerate(dataloader):
                # get inputs
                inputs, labels = data
                #inputs = inputs.view(28 * 28 * 1, -1)

                optimizer.zero_grad()

                # Forward + Backward + Optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward() # computes the gradients
                optimizer.step() # adapt the weights

    elif mode == 'test':
        model.eval()

        # prediction on unseen data
        correct, total = 0, 0

        for i, data in enumerate(dataloader):
            inputs, labels = data
            #inputs = inputs.view(-1, 28 * 28 * 1)
            outputs = model(inputs) # calculates scores for each class
            _, outputs = torch.max(outputs.data, 1) # get class with highest score
            #preds.append(outputs)
            total += labels.size(0)
            correct += (outputs == labels).sum().item()

        print('Test accuracy: %d %%' %(100 * correct / total))
    


def readLabels(data_dir = './data'):

    with open(data_dir + '/cifar/labels.txt') as label_file:
        labels = label_file.read().split()
        #print(labels)
        label_mapping = dict(zip(labels, list(range(len(labels)))))

    return label_mapping

def readData(data_dir = './data'):

    train_dir = data_dir + '/cifar/train'
    test_dir = data_dir + '/cifar/test'

    return train, test


if __name__ == "__main__":

    # TODO maybe add normalization with STD and MEAN
    transform = transforms.Compose([transforms.ToTensor()])

    trainset = MyCIFARDataset(data_dir = './data/cifar/train', transforms = transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size = 128, shuffle = True, num_workers = 1)

    testset = MyCIFARDataset(data_dir = './data/cifar/test', transforms = transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size = 128, shuffle = True, num_workers = 1)    

    train_iter = iter(train_loader)
    #print(type(train_iter))

    images, labels = train_iter.next() # 3 channels per picture (32 x 32), 128 per batch
    print(images.size(), labels.size())

    # TRAINING
    myCNN = MyCNN();
    optim = optim.SGD(myCNN.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    runCNN(model = myCNN, mode = 'train',
        criterion = nn.CrossEntropyLoss(),
        optimizer = optim,
        dataloader = train_loader, epochs = 5)

    # TEST
    runCNN(model = myCNN, mode = 'test',
        criterion = nn.CrossEntropyLoss(),
        optimizer = optim,
        dataloader = test_loader)
