# Electroencephalography (EEG) data classification for Brain Computer Interface (BCI).

## Introduction

Motor Imagery Brain-Computer Interface (BCI) is a technology that allows individuals to control
external devices using their brain signals. It works by detecting patterns in the brain activity of a
person while they imagine performing a particular movement, such as moving their arm or leg. These
patterns are then translated into computer commands that can be used to control a device.
Deep learning is a powerful technique that has shown promise in improving the accuracy and
efficiency of Motor Imagery BCIs. Deep learning algorithms can automatically learn to extract
relevant features from raw brain signals, which can be used to classify different types of movements.
This can help to overcome some of the challenges associated with traditional machine learning
approaches, which often require manual feature selection and engineering.
We propose the following topic for our project: Build a novel deep learning pipeline to increase the
accuracy in motor intent classification compared to existing literature. This project is an application
project with the possible directions of:

1. data augmentation: either find and incorporate more publicly available dataset or think of ways to
do data augmentation for better training.

2. data pre-processing of raw EEG data: we could implement noise removal to obtain better quality
data.

3. selection of model architectures: Play with model architectures (CNN, RNN, transformers).

4. transfer learning : current field of BCI is challenged by models not transferable to make predictions
on new subjects? F Find ways to improve this.

## Database Description

The database for this project contains EEG data from 37 healthy individuals who participated in a
brain-computer interface (BCI) study. All but one subject underwent 2 sessions of BCI
experiments that involved controlling a computer cursor to move in one-dimensional space using
their “intent”. EEG data were recorded with 62 electrodes. In addition to the EEG data, behavioral
data including the online success rate and results of BCI cursor control are also included.
For each subject, 250 trials of EEG data were recorded at each electrode at 1000 Hz with a duration of
6 seconds (6000x62 data points per trial). Each trial is labeled numerically with 1 or 2 that represent
left hand imagination and right hand imagination, respectively.

## Baseline Models

The CNN model architecture used in [2] was derived from the “shallow CNN” proposed by [4]. This
architecture consists of two convolution layers which convolve the temporal and spatial dimensions
respectively. Then, there is a single mean pooling layer followed by a classification layer which
consists of a dense linear layer and a Softmax layer. It is noteworthy that this architecture explicitly
attempted to replicate the behaviors of the Filter bank common spatial patterns (FBCSP) pipeline,
which was the state-of-the-art EEG signal interpretation technique at the time. The CNN models
were trained using the brain_decode package (version 0.4.85) in Python.

## Midterm Milestone

1. Perform data cleaning on the our selected dataset.
2. Replicate the CNN pipeline in baseline paper for motor intent classfication.
3. Achieve similar level of accuracy as reported in the baseline paper
