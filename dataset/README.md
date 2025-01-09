# Dataset

## Update

In the journal version, we have added exepriemnts on Graph-based dataset. 

The Graph-based ST data can be obtained in [METR-LA](https://zenodo.org/records/5146275).

## Download

To access the dataset, follow these steps:

1. Click on the following link to download the zip file:

- Google Drive: [Data Download Link](https://drive.google.com/drive/folders/1jiTLlOgc0kzwmM12q9M6OOrBSXVD_kUU?usp=sharing).
- Baidu Netdisk: [Data Download Link](https://pan.baidu.com/s/1G8MZj5Jn9akQGb78QRFzHw?pwd=4646).
- Tsinghua Cloud: [Data Download Link](https://cloud.tsinghua.edu.cn/d/87f5954d4f6f4ebd9d70/)

2. Please create a folder named ``dataset`` within your UniST project directory.

3. After downloading the dataset files, move them to the following path ``UniST/dataset``.


## Data Structure Overview

This repository contains JSON data. The dataset is organized into training, validation, and testing sets. Below is an overview of the data structure and key components:

### Structure
The data is stored in a dictionary with the following keys:

- `X_train`: Training dataset
- `X_test`: Testing dataset
- `X_val`: Validation dataset
- `timestamps`: Contains timestamp information, including the day of the week and time of day.

### Data Dimensions

**1. Spatio-temporal Data**
   - **Key**: `X_train`, `X_test`, `X_val`
   - **Shape**: $\(N \times T \times H \times W\)$
   - **Description**:
     - $\(N\)$: Number of samples
     - $\(T\)$: Temporal length
     - $\(H\)$: Height of the spatial grid
     - $\(W\)$: Width of the spatial grid

**2. Periodical Data**
   - **Key**: Included in `X_train`, `X_test`, `X_val` under periodical sections
   - **Shape**: $\(N \times T \times P \times H \times W\)$
   - **Description**:
     - $\(P\)$: Number of past days corresponding to the time of day

Each dataset key (`X_train`, `X_test`, `X_val`) contains both spatio-temporal and periodical data organized accordingly.

