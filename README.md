# Axial-Factorized Attention: A Fast and Efficient Approach to Real-Time Semantic Segmentation for MCU Deployment

## Environment
python = 3.8.20

CUDA = 12.2

`conda env create --file AFA.yaml`

## Requirement
Download the **Cityscapes dataset** and set `data_root` in `main.py`.

## Run


| Model Name | Performance (mIoU)  | 
|--------------|------------------| 
|  **AFASeg_S** | **72.8** | 
|  **AFASeg_XS** | **71.0** | 
|  **AFASeg_XXS** | **69.5** |


`bash run.sh [model_name]`
