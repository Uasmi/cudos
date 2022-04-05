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
In this example, we will go through the process and install Tensorflow 1.12.0.  
We will start by getting the correct version of CUDA drivers from the NVIDIA website.

>You'll need a developer account for getting the drivers.  
>You can register the developer account at this [link](https://developer.nvidia.com/login)

From the Tensorflow compatibility table, we can see that we need to download CUDA 9.0 and cuDNN 7:
Version | Python Version | GCC | cuDNN | CUDA
--- | --- | --- | --- | --- |
tensorflow-gpu-1.12.0 | python 3.3-3.6 | GCC 4.8 | 7.9.5 | 9.0
>Tip: you can grab the latest cuDNN version available for specific CUDA installation.

The archived versions are available at:  
[CUDA](https://developer.nvidia.com/cuda-toolkit-archive)  
[cuDNN](https://developer.nvidia.com/rdp/cudnn-archive)


