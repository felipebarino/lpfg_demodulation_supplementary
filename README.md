# FBG Sensor Array and Self-Attention AI Model for Cost-Effective Demodulation of Long-Period Fiber Grating Sensor

## Overview

This repository contains supplementary materials for the research paper 'FBG Sensor Array and Self-Attention AI Model for Cost-Effective Demodulation of Long-Period Fiber Grating Sensor' by Felipe Oliveira Barino and Alexandre Bessa dos Santos. The paper proposes a novel and cost-effective method for long-period fiber grating (LPFG) demodulation using a combination of readily available fiber Bragg grating (FBG) sensor arrays and a self-attention AI model. This method eliminates the need for expensive or complex optoelectronics and uses a quasi-distributed sensor network, simplifying integration into industrial settings.

## Repository Structure

- `req.txt`: Contains the requirements to run this repository.

- `data/`: This directory contains the measurements and supporting data for our experiments.

- `ml_model/`: This directory contains the machine learning model (`model.h5`), its weights after model selection (`model_weights.h5`) and after it was fine tunned(`model_weights_finetunned.h5`).

- `model_fcn.py`: Contains functions functions related to the ML model, they were used to train, evaluate, preprocess and load data.

- `utils.py`: Contains utility functions that are likely used in other parts of the codebase.

- `1 - Synthetic data generation.ipynb`: This notebook details the process of generating synthetic long-period fiber grating (LPFG) spectra..

- `2 - Exploring the model.ipynb`: This notebook focuses on exploring the self-attention AI model used for LPFG demodulation..

- `3 - Evaluation by measured spectra.ipynb`: This notebook describes the evaluation of the self-attention AI model using measured LPFG spectra.

## The notebooks

### 1 - Synthetic data generation.txt

* Importing necessary packages, such as NumPy for numerical computation and Matplotlib for plotting.

* Defining the parameters for generating the LPFG spectra, including the desired resonant wavelength, coupling strength, full width at half maximum (FWHM), insertion loss, and a function number to select the type of function used to model the LPFG response (in this case, a Lorentzian function).

* Generating the synthetic LPFG spectra using the specified parameters. This involves creating both "clean" (ideal) spectra and "noisy" spectra, where noise is simulated by adding extra attenuation bands with random parameters.

* Visualizing the generated spectra, including plotting examples of LPFG spectra with different resonant wavelengths and coupling efficiencies.

### 2 - Exploring the model.txt

* Importing necessary packages, including TensorFlow and Keras for deep learning, as well as other packages for data manipulation and visualization.

* Loading the pre-trained self-attention AI model and its fine-tuned weights.

* Visualizing the model's architecture, showing the layers, their connections, and activation functions.

* Displaying the model's weights, which are the parameters learned during training. These weights are visualized as heatmaps to understand their distribution and impact on the model's predictions.

* Analyzing the layer outputs as a function of different LPFG parameters, such as resonant wavelength, coupling efficiency, and FWHM. This involves feeding synthetic LPFG spectra with varying parameters to the model and observing the activations of different neurons in each layer. This analysis helps in understanding how the model processes the input data and identifies the relevant features for accurate LPFG demodulation.

### 3 - Evaluation by measured spectra.txt

* Importing necessary packages for loading the model, data manipulation, statistical analysis, and visualization.

* Loading the measured LPFG spectra obtained from experiments. These spectra may have variations and noise due to real-world conditions.

* Loading the same measured spectra preprocessed by simulating the effect of fiber Bragg grating (FBG) arrays.

* Evaluating the performance of the proposed self-attention AI model on the measured spectra. This involves comparing the model's predictions of the resonant wavelengths with the actual resonant wavelengths of the measured spectra.

* Making predictions with uncertainty estimation using the dropout technique. This involves running the model multiple times with different dropout configurations to estimate the uncertainty in the model's predictions.

* Evaluating the performance of a baseline model, which in this case is the Lorentzian fitted model, on the same measured spectra. This provides a benchmark for comparing the performance of the proposed AI model.

* Comparing the performance of the proposed model and the baseline model using various metrics, such as root mean squared error (RMSE), mean absolute percentage error (MAPE), and R-squared (coefficient of determination).

* Conducting statistical tests to determine if there is a significant difference in the performance of the two models.

* Analyzing specific cases where the Lorentzian fitted model shows poor performance and explaining the reasons for the discrepancies.

* Discussing the role of the attention mechanism in improving the model's performance, especially in handling spectra with distortions and noise.

* Evaluating the performance of the LPFG sensor for refractive index (RI) sensing. This involves characterizing the sensor's response to different concentrations of glycerol/water mixtures, which have varying refractive indices.

## Usage

The cells outputs can be seen as is at this repository. These represent the data as subimitted/published.

Please note that given the stochastic nature of this proposal, results might slightly vary.

To run, the Jupyter notebooks can be run in any Python environment that meets the requirements (Python 3.10.9 and req.txt file).

## Contact

If you have any questions, please contact Felipe Oliveira Barino at felipebarino@gmail.com