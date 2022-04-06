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
In this example, we will go through the process and install Tensorflow 1.13.1.  
We will start by getting the correct version of CUDA drivers from the NVIDIA website.

>You'll need a developer account to get the drivers.  
>You can register the developer account at this [link](https://developer.nvidia.com/login)

From the Tensorflow compatibility table, we can see that we need to download CUDA 10.0 and cuDNN 7.4:
Version | Python Version | GCC | cuDNN | CUDA
--- | --- | --- | --- | --- |
tensorflow_gpu-1.13.1 | python 3.3-3.8 | GCC 4.8 | 7.4 | 10.0

The archived versions are available at:  
[CUDA](https://developer.nvidia.com/cuda-toolkit-archive)  
[cuDNN](https://developer.nvidia.com/rdp/cudnn-archive)

### CUDA
CUDA Toolkit provides a development environment for creating high-performance GPU-accelerated applications.
It is the first thing we need to install, so go to the [CUDA-10.0 Archive Page](https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=deblocal) and grab a local deb file:

And then run the following commands in terminal:
```shell
sudo dpkg -i cuda-repo-ubuntu1804-10-0-local-10.0.130-410.48_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-10-0-local-10.0.130-410.48/7fa2af80.pub
sudo apt-get update
```
>Note: remember to enter your CUDA version in the <version> bracket, if you're following NVIDIA documentation.
  
We also want to add CUDA version to the last console command:
```
sudo apt-get install cuda-10.0
```

That's it for the CUDA part. Now let's jump to cuDNN.

### cuDNN
  



