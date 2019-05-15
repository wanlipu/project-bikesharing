# Project Bikesharing

This is part of Udacity Deep Learning Program, and this repository is meant to document my progress towards tv bikesharing project completion.

Predicting Bike-Sharing Patterns: construct a neural network from scratch in NumPy to predict bike rentals.

## Steps
- Forward Pass
- Backward Pass
- Hyperparameter Tuning

## Dependencies

### Configure and Manage Project Environment with Anaconda

Per the Anaconda [docs](http://conda.pydata.org/docs):

> Conda is an open source package management system and environment management system 
for installing multiple versions of software packages and their dependencies and 
switching easily between them. It works on Linux, OS X and Windows, and was created 
for Python programs but can package and distribute any software.

### Overview
Using Anaconda consists of the following:

1. Install [`miniconda`](http://conda.pydata.org/miniconda.html) on your computer, by selecting the latest Python version for your operating system. If you already have `conda` or `miniconda` installed, you should be able to skip this step and move on to step 2. You can also find achived miniconda version in this [`link`](https://repo.continuum.io/miniconda/), I used Miniconda3-4.5.1-Windows-x86_64 (python 3.6.5)
2. Create and activate * a new `conda` [environment](http://conda.pydata.org/docs/using/envs.html).

- __Windows__: 
```
conda create --name deep-learning python=3.6.5
conda activate deep-learning
conda install numpy matplotlib pandas jupyter notebook
```
