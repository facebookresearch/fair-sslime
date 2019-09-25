# Installation

Our installation is simple and anaconda3 based. Follow the steps below:

**Requirements**: NVIDIA GPU, Linux

**Note:** We currently do not provide support for CPU only runs except SVM trainings.


## Step 1: Install Anaconda3

```bash
cd $HOME
wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh -O anaconda3.sh
chmod +x anaconda3.sh
./anaconda3.sh -b -p $HOME/anaconda3
rm anaconda3.sh
```

Now add anaconda3 to your PATH so that you can use it. For that run the following command:

```bash
export PATH=$HOME/anaconda3/bin:$PATH
```

Now, verify your conda installation and check the version:

```bash
which conda
```

This command should print the path of your conda bin. If it doesn't, make sure conda is in your $PATH.

Now, let's create a conda environment which we will work in.

```bash
conda create --name ssl-framework python=3.6
source activate ssl-framework
```

## Step 2: Install dependencies using conda

We use several conda packages that are installed easily as below:

```bash
conda install -c pytorch pytorch
conda install -yq future protobuf pyyaml six scipy pycurl opencv scikit-learn cython networkx
# To test the installation works:
python -c 'import torch'
```

## Step 3: FAIR SSLIME

Now clone this repository and install using instructions:

```bash
cd $HOME && git clone --recursive https://github.com/facebookresearch/fair-sslime.git
python setup.py install
# To test the installation works
python -c 'import sslime'

```

That's it! You are now ready to use this code.
