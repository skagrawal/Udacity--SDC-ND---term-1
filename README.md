# Udacity--SDC-ND---term-1
This repository includes projects done as Self-Driving Car Engineer Nanodegree term 1 by Udacity.

Project 1 - Finding Lane Lines on the Road using computer vision

Project 2 - Traffic Sign Classifier using deep learning and TensorFlow

Project 3 - Behavioral Cloning using deep learning and Keras

Project 4 - Advanced Lane Finding using computer vision

Project 5 - Vehicle Detection and Tracking using machine learning and computer vision



### Setup
There are two ways to get up and running:

## [Anaconda Environment](doc/configure_via_anaconda.md)

Get started [here](doc/configure_via_anaconda.md). More info [here](http://conda.pydata.org/docs/).

Supported Sytems: Linux (CPU), Mac (CPU), Windows (CPU)     

| Pros                         | Cons                                               |
|------------------------------|----------------------------------------------------|
| More straight-forward to use | AWS or GPU support is not built in (have to do this yourself)              |
| More community support       | Implementation is local and OS specific            |
| More heavily adopted         |                                                    |

## [Docker](doc/configure_via_docker.md)

Get started [here](doc/configure_via_docker.md). More info [here](http://docker.com).

Supported Systems : AWS (CPU, [GPU](doc/docker_for_aws.md)), Linux (CPU), Mac (CPU), Windows (CPU)     

| Pros                                | Cons                                 |
|-------------------------------------|--------------------------------------|
| Configure once for all environments | More challenging to use              |
| AWS, GPU support                    | Less community support               |
| Practice with Docker              | Have to manage images and containers |
|                                     |                                      |


### Dependencies
This project requires **Python 3.5** and the following Python libraries installed:

- [Jupyter](http://jupyter.org/)
- [NumPy](http://www.numpy.org/)
- [SciPy](https://www.scipy.org/)
- [scikit-learn](http://scikit-learn.org/)
- [TensorFlow](http://tensorflow.org)
- [Matplotlib](http://matplotlib.org/)
- [Pandas](http://pandas.pydata.org/) (Optional)

Run this command at the terminal prompt to install [OpenCV](http://opencv.org/). Useful for image processing:

- `conda install -c https://conda.anaconda.org/menpo opencv3`
