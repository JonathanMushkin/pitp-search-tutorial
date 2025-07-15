# Gravitational Wave Search Tutorial

This assignment is the first part of two exercises, in which we will analyze LIGO data to find the gravitational wave transient caused by the coalescence of two neutron stars (GW170817).

## Overview

You are given a true 2048-second segment of Hanford LIGO data, sampled at 4096 Hz (down-sampled from the original 16 kHz data for convenience). The data used in this tutorial is public GWOSC (Gravitational Wave Open Science Center) data that has been processed and saved at a lower sampling resolution to make it more manageable for educational purposes.

## What's Included

1. `strain.npy` - NumPy-readable file containing the strain data
2. `gw_search_functions.py` - Helper functions, constants, and function skeletons for completion
3. `pitp-gw-search-1.ipynb` - Main tutorial notebook
4. `pyproject.toml` - Environment configuration file

Note: The timestamps corresponding to the strain are not uploaded due to size constraints and are instead provided in `gw_search_functions`.

## Learning Objectives

In this notebook, we will practice:

- Using FFT/RFFT to perform matched-filter searches in gravitational wave data
- Computing test statistics for signal detection
- Understanding the **normalization** of inputs and expected test statistic values
- Applying glitch-removal procedures to real LIGO data

## Mathematical Background

Under the null and signal hypotheses, the data model is:

$$H_0: \quad s(t) = n(t)$$
$$H_1: \quad s(t) = n(t) + h(t)$$

The noise $n(t)$ is approximately stationary and Gaussian with a certain power spectral density $S_n(f)$. Under the Gaussian noise approximation, the log-likelihood of waveform $h$ given strain data $s$ is:

$$\ln \mathcal{L} = \Re \langle h, s \rangle - \frac{1}{2} \langle h, h \rangle$$

with the inner product defined as:

$$\langle a, b \rangle = \sum_f \frac{a(f) b^\ast(f)}{S_n(f)}\,\mathrm{d}f = \sum_f \tilde{a}(f) \tilde{b}^\ast(f)\,\mathrm{d}f$$

where the tilde denotes the whitened series.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/JonathanMushkin/GW_search_tutorial.git
cd GW_search_tutorial
```

### 2. Create Environment Using pyproject.toml

The repository includes a `pyproject.toml` file that defines all necessary dependencies. You can use it to create a conda environment:

#### Option A: Using conda/mamba (recommended)

```bash
# Create environment from pyproject.toml
conda env create -f environment.yml
conda activate gw_detection_tutorial
```

#### Option B: Using pip with conda

```bash
# Create a new conda environment
conda create -n gw_detection_tutorial python=3.9
conda activate gw_detection_tutorial

# Install dependencies
pip install -e .
```

#### Option C: Using pip only

```bash
# Create virtual environment
python -m venv gw_env
source gw_env/bin/activate  # On Windows: gw_env\Scripts\activate

# Install dependencies
pip install -e .
```

### 3. Launch Jupyter Notebook

```bash
jupyter lab pitp-gw-search-1.ipynb
```

or

```bash
jupyter notebook pitp-gw-search-1.ipynb
```

## Repository Structure

```
GW_search_tutorial/
├── README.md                    # This file
├── pyproject.toml              # Environment and dependency configuration
├── pitp-gw-search-1.ipynb      # Main tutorial notebook
├── gw_search_functions.py      # Helper functions and utilities
├── strain.npy                  # LIGO strain data (downsampled)
├── notebooks/                  # Additional notebook materials
└── solution/                   # Solution materials
```

## Data Information

The strain data (`strain.npy`) contains:
- **Duration**: 2048 seconds of LIGO Hanford data
- **Sampling Rate**: 4096 Hz (downsampled from original 16 kHz)
- **Event**: Contains the GW170817 gravitational wave signal
- **Source**: Public GWOSC data, processed for educational use

## Course Information

This exercise is based on an assignment given in the course "Statistics, Algorithms and Experiment Design" at the Weizmann Institute of Science by:

- **Lecturer**: Barak Zackay
- **Teaching Assistants**: Ariel Perera, Dotan Gazith, Jonathan Mushkin, and Oryna Ivashtenko

## Contact

For any help, questions, or comments, please contact:
**jonathan.mushkin[at]weizmann.ac.il**

## Acknowledgments

This tutorial uses publicly available gravitational wave data from the LIGO Scientific Collaboration via the Gravitational Wave Open Science Center (GWOSC). 