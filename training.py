#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing all the important and reuqired libraries

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
from matplotlib import pyplot as plt
from project1_model import project1_model

# for data augmentation - used RandomHorizontalFlip, RandomCrop, Normalize for train_data 
# and Normalize for test_data

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(size=32, padding=[0, 2, 3, 4]),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# setting batch size to be 64
batch_size = 64

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

net = project1_model()
if torch.cuda.is_available():
    net.cuda()

def train_test_model():
    # using cuda if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    # defining the loss and optimizer
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    #declaring our losses
    test_loss_history = []
    train_loss_history = []
    acc_history_train = []
    acc_history_test = []

    # training and testing for 200 epochs
    for epoch in range(200):

        train_loss,test_loss = 0.0,0.0
        correct_train,correct_test,total_train,total_test = 0,0,0,0

        for i, data in enumerate(trainloader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            predicted_output = net(images)
            fit = loss(predicted_output,labels)
            fit.backward()
            optimizer.step()
            train_loss += fit.item()
            _, predicted_output = torch.max(predicted_output.data, 1)
            correct_train += (predicted_output == labels).sum().item()
            total_train += labels.size(0)

        for i, data in enumerate(testloader):
            with torch.no_grad():
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                predicted_output = net(images)
                fit = loss(predicted_output,labels)
                test_loss += fit.item() 
                _, predicted_output = torch.max(predicted_output.data, 1)
                correct_test += (predicted_output == labels).sum().item()
                total_test += labels.size(0)

        train_loss = train_loss/len(trainloader)
        test_loss = test_loss/len(testloader)
        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)

        train_accuracy = (correct_train*100)/total_train
        test_accuracy = (correct_test*100)/total_test
        acc_history_train.append(train_accuracy)
        acc_history_test.append(test_accuracy)


        print('Epoch {0}, Train loss {1}, Train Accuracy {2}, Test loss {3}, Test Accuracy {4}'.format(
            epoch, train_loss, train_accuracy, test_loss, test_accuracy))

    # plotting the Test/Train accuracy vs Epoch
    def plot_graphs_accuracy(train_accuracy=[], test_accuracy=[]):
        plt.plot(acc_history_train)
        plt.plot(acc_history_test)
        max_acc_test = max(acc_history_test)
        max_accuracy_test_index = acc_history_test.index(max_acc_test)
        plt.plot(max_accuracy_test_index, max_acc_test, '.')
        plt.text(max_accuracy_test_index, max_acc_test, " Max Accuracy = {0}".format(
        max_acc_test))
        plt.legend(["train", "test"])
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.grid()
        plt.show()

    # plotting the Test/Train losses vs Epoch
    def plot_graphs_losses(train_loss_history=[], test_loss_history=[]):

        plt.plot(train_loss_history)
        plt.plot(test_loss_history)
        plt.legend(["train", "test"])
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.title('Test/Train Loss Vs Epoch')
        plt.grid()
        plt.show()

    #calling both the above plotting functions    
    plot_graphs_accuracy(train_accuracy=acc_history_train,test_accuracy=acc_history_test)
    plot_graphs_losses(train_loss_history=train_loss_history, test_loss_history=test_loss_history)


    # Calculating the number parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('The Number of Parameters are: {0}'.format(count_parameters(net)))

    # saving the weights in .pt file
    model_path = './project1_model.pt'
    torch.save(net.state_dict(), model_path)

train_test_model()


# In[2]:





# In[ ]:




