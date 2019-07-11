# DHDN_keras
DHDN(Densely Connected Hierarchical Network for Image Denoising) and DIDN(Deep Iterative Down-Up CNN for Image Denoising), Un-Official implementation  

# dependency
- python 3.6  
- keras 2  
- tensorflow 1.12+  
- Linux (if run in Windows, maybe need modify all path string, '/' to '\\')  
  
# train and test
`cd code`
## train
1. `vi train.py` modify training configuration and set path(data_path, load_model_path)
2. `python train.py`

## test
`python test.py`