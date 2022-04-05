# Cudo's
Managing Multiple Cuda Versions on your system.

## Overview
Training and developing machine learning models require using various frameworks installed on your computer. Sometimes it results in multiple driver versions in your environments, which can become a hassle to match and debug.

While we can still use Docker to create a separate image for each framework, there's a way to manage CUDA/cuDNN versions using only Linux shell and Conda (if required or you feel unfamiliar with Docker).

## Getting the correct version
If you're working with a code based on the old framework version, you must check what driver version you need to install.
Depending on the framework you are using (ex. Tensorflow, PyTorch), you'll need to install a proper CUDA, cuDNN, and GCC version.  
Below you can see compatibility tables for:

[Tensorflow](https://www.tensorflow.org/install/source#gpu)  
[PyTorch](https://pytorch.org/get-started/previous-versions/)


Another thing to consider is CUDA and cuDNN matching, which you can check in the official NVIDIA documentation:
https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html


>Tip: Tensorflow compatibility table includes both  
>CUDA, cuDNN, and GCC versions, which is pretty usable!


## Installing the drivers
### CUDA, cuDNN
We will start by getting the correct version of CUDA drivers from the NVIDIA website.  
In this example, we will go through the process and install Tensorflow 1.13.1.


>You'll need a developer account for getting the drivers.  
>You can register the developer account at this link: 

