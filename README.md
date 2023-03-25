# Sleep vs Nirodha: EEG Signal Classification using EEGNet Deep Neural Network

This GitHub repository contains the code for a project that involves working with EEG signals to distinguish between sleep and nirodha states. The project focuses on using a specific preprocessing pipeline that involves four key stages, namely:

1. Baseline Correction - This stage involves adjusting the signal baseline to zero or a defined level, which helps to remove any unwanted DC offset.

2. Re-referencing - This stage involves changing the reference electrodes to reduce the influence of external noise and enhance the signal quality.

3. DC offset - This stage involves removing the direct current (DC) component of the signal, which can affect the signal's amplitude and make it difficult to analyze.

4. ICA-based muscle artifact removal - This stage involves using Independent Component Analysis (ICA) to identify and remove any muscle artifacts in the signal, which can be caused by movements or muscle contractions.

In addition to this robust preprocessing pipeline, the project leverages the state-of-the-art EEGNet deep neural network to perform a binary classification of these physiological states of mind. EEGNet is a deep convolutional neural network (CNN) that has been specifically designed for EEG signal classification. The network architecture includes a depthwise separable convolution, which can capture both spatial and temporal features of the EEG signal.

The repository includes the following files:

1. `Code`: This file contains the code for the four-stage preprocessing pipeline and the EEGNet model predictions.

2. `Data`: This file contains the raw and preprocessed eeg file data.

3. `README.md`: This file provides an overview of the project and its contents.
