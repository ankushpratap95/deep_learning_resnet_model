# Deep Learning Resnet Model

By [Ankush Pratap Singh](), [Dhruv Agarwal](), [Kirk Nickish]()

### Table of Contents
0. [Introduction](#introduction)
0. [Disclaimer](#disclaimer)
0. [Architecture](#Architecture)
0. [Model Training](#ModelTraining)
0. [Result](#Result)
0. [How to run the model](#How_to_run_the_model)


### Introduction

This repository contains the proposed modified model of Resnet, with 4.9 M parameters. In this repo we have trained our model on different optimizers like ADAM, SGD, RMSProp and Adagrad models and chose ADAM as best for the CIFAR-10 dataset.

0. In this repo there are 3 files:- 
 - .ipynb file contains the script of our model which can be simply run.
 - .pt file contains the trained weights of our resnet architecture.
 - .pdf file is the report with detailed explanation and results of our model. 

### Disclaimer 

0. If you want to train these models, please notice that:
	- GPU establishment is required. 
	- Changes of mini-batch size should impact accuracy (we use a mini-batch of 64 images).
	- We have implemented data augmentation, dropout layers and normalization techniques for best accuracy. 
    - We are running these models on 200 epochs at 0.01 learning rate. 

### Architecture 

0. Original Resnet-18 model provides 11 Million parameters for the residual blocks [2,2,2,2].

 - We have run the model on [3,3,3,3] and [1,1,1,1], in this we got about 17 million parameters for [3,3,3,3] and 4.9 million parameters for [1,1,1,1]. 
 - We kept number of input channels to be 64, kernel size to be 3x3 and average pool size to be 4x4. 
 - Other parameters remains unchanged for resnet model. 


### Model_Training

0. In training of this model, we have used different optimizers like ADAM, SGD, RMSProp and Adagrad. After comparing them Adam came out to be best among all. 
 - 91% of accuracy has been observed by Adam Optimizer. 

0. Learning Rate- We have observed accuracy on different learning rate such as 0.5, 0.1  and 0.01. In lr of 0.01 we achieved the maximum accuracy.

0. Implemented Data augmentation such as random horizontal flip, random crop and normalization techniques in our model.

0. In regularization, we have added Dropout layers for probability of 0.1 after observing it from 0.1 to 0.5 as p=0.1 gives best accuracy.



### Result

After running our model on 200 epochs we got the maximum testing accuracy of about 91%, with a training loss of about 0.02 and testing Loss of about 0.71. 


### How_to_run_the_model (GPU required)

0. Download or clone the required repo and then
 - 1st Method- Directly run the .ipynb file on google colab or jupyter notebook.
 - 2nd Method- Load .pt file and run it accordingly on your dataset. 
