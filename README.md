# Cudo's - Managing Multiple Cuda Versions on your system.

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

![](https://github.com/Uasmi/cudos/blob/main/pictures/cuda-10.0.png?raw=true)

Now switch to the terminal and run the following commands:
```shell
sudo dpkg -i cuda-repo-ubuntu1804-10-0-local-10.0.130-410.48_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-10-0-local-10.0.130-410.48/7fa2af80.pub
sudo apt-get update
```
>Note: remember to enter your CUDA version in the <version> bracket, if you're following NVIDIA documentation.
  
We also want to modify the last command from the official documentation so that it will include a specific CUDA version:

```shell
sudo apt-get install cuda-10.0
```
You can verify the installation by running 

```shell
cd /usr/local
ls
```
You should see a folder named ```cuda-10.0```.

That's it for the CUDA part. Now let's jump to cuDNN.

### cuDNN
The NVIDIA CUDA Deep Neural Network library (cuDNN) is a GPU-accelerated library of primitives for deep neural networks. 
In our case, we need a **7.4 cuDNN version**. Go to [cuDNN Archive Page](https://developer.nvidia.com/rdp/cudnn-archive), scroll down until you'll see **cuDNN v7.4.2 for CUDA 10.0**, and click on the cuDNN Library for Linux:
![](https://user-images.githubusercontent.com/14073415/161923701-b2540046-9bdc-4ecd-832d-6dc65defe2bf.png)

Go to the terminal and install cuDNN by running these commands:
```shell
tar -xzvf cudnn-10.0-linux-x64-v7.4.2.24.tgz
sudo cp cuda/include/cudnn*.h /usr/local/cuda-10.0/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda-10.0/lib64

sudo chmod a+r /usr/local/cuda-10.0/include/cudnn*.h /usr/local/cuda-10.0/lib64/libcudnn*
```

_Optional: If you're installing other CUDA and cuDNN versions, run the code below and replace the <> bracket with the required version._
```shell
tar -xzvf cudnn-<>-linux-x64-v<>.tgz
sudo cp cuda/include/cudnn*.h /usr/local/cuda-<>/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda-<>/lib64

sudo chmod a+r /usr/local/cuda-<>/include/cudnn*.h /usr/local/cuda-<>/lib64/libcudnn*
```

After that, you can verify installation by running ```nvcc --version```
>Tip: you can find the full installation guide [here](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html).
  
### GCC
Each version of the CUDA driver has its GCC compiler compatibility.
Here you can see the highest supported GCC version for each CUDA version.

CUDA version |	Supported highest GCC version
| --- | --- |
11.4.1+, 11.5, 11.6 | 11
11.1, 11.2, 11.3, 11.4.0 | 10
11 | 9
10.1, 10.2 | 8
9.2, 10.0 | 7
9.0, 9.1 | 6
8 | 5.3
7 | 4.9
5.5, 6 | 4.8
4.2, 5 | 4.6
4.1 | 4.5
4.0 | 4.4

You can check your current GCC version by running: ```gcc -v```
 
>Tip: you can run ```dpkg --list | grep compiler``` to see if you have a correct GCC compiler installed.

We can see that our version is higher than the supported GCC version, so let's install the proper one:
```shell
sudo apt install gcc-7 g++-7
```
  
In order for CUDA to choose a proper GCC compiler, it is enough to create a softlink between CUDA and GCC:
```shell
sudo ln -s /usr/bin/gcc-7 /usr/local/cuda-10.0/bin/gcc 
sudo ln -s /usr/bin/g++-7 /usr/local/cuda-10.0/bin/g++
```

## Managing multiple CUDA versions
### Pre-work
Before jumping into the last steps of this guide, it is a good time to install all the required CUDA versions on your system (you can repeat the steps above for each version of CUDA). 
After doing so, you can get a list of all CUDA versions by running this command:
```shell
cd /var/local
ls
```
  
You will see a full list of available CUDA drivers as folders like this:
![](https://user-images.githubusercontent.com/14073415/161936368-aa65c2a1-890b-4ab8-af0e-3e7c6997c59d.png)

  
### Setting up CUDA path variable
To make our Conda environments choose the correct CUDA version, we need to modify **bashrc** file:
```shell
sudo gedit ~/.bashrc
```
 
Now scroll to the bottom of the file and add:
```shell
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}

export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
```
  
You should add all the required CUDA versions to this file, so in the end, it will look something like this:
```shell
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}

export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda-10.0/lib64:/usr/local/cuda-10.1/lib64:/usr/local/cuda-11.2/lib64:/usr/local/cuda-11/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
```

Save the file and restart your computer.
  
### Verifying installation
Now, let's run nvcc --version to check the default CUDA version on our system:
![](https://user-images.githubusercontent.com/14073415/161940141-588525d2-a394-472f-b2b2-e44c232ffeae.png)

In my case, you can see that the version is 11.0.

Let's verify that everything works correctly by creating a new Conda environment with 1.13.1 version of Tensorflow:
```shell
conda create -n tensorflow-1.13 python=3.6
conda activate tensorflow-1.13
conda install tensorflow-gpu==1.13.1
python -c 'import tensorflow as tf; print(tf.__version__); print(tf.test.gpu_device_name()); print(tf.test.is_gpu_available())'
```
You will see Tensorflow version information as well as available GPUs.

Let's check if we have a proper CUDA driver selected, by running:
```shell
conda list cuda
```
![](https://github.com/Uasmi/cudos/blob/main/pictures/cuda-verified.png?raw=true)
 
Thats it! You are now free to use any combination of CUDA/cuDNN drivers for various frameworks you want.
  
Feel free to post an Issue if you encounter any problem.
