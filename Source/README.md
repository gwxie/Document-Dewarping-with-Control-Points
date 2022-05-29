# Description
Due to work, more information will be released.

# Performance
**Execution Time when getting a 1024x960 corrected image.**
|   Method  | Time  |
|  ----  | ----  |
| TPS  | - |
| Interpolation  | - |
<!-- 
<small>\* TPS:flatByfiducial_TPS,
  
  Interpolation:flatByfiducial_interpolation</small> -->

# Testing
1、Download model parameter and source codes 

2、Resize the input image into 992x992

3、Running 

- In GPU 0:
`python test.py --data_path_test=./your/test/data/path/ --parallel 0 --schema test --batch_size 1`

- In CPU:
`python test.py --data_path_test=./your/test/data/path/ --parallel None --schema test --batch_size 1`

# Training
a) Download training data in [here](https://github.com/gwxie/Document-Dewarping-with-Control-Points/tree/main/Source/dataset/fiducial1024).

b) Run `python train.py --data_path_train=./your/train/data/path/ --data_path_validate=./your/validate/data/path/ --data_path_test=./your/test/data/path/ --batch_size 32 --schema train --parallel 01`

# Use your Dataset
The training dataset can be synthesised using the [scripts](https://github.com/gwxie/Synthesize-Distorted-Image-and-Its-Control-Points).

# Q&A
1. Q:How to adjust the output image resolution？
A: Adjusting the ‘flat_shap’ or 'output_shape' in utilsV4.py.
https://github.com/gwxie/Document-Dewarping-with-Control-Points/blob/4c74853b0eb93f7c6006a774c2eb42c64f363531/Source/utilsV4.py#L99
or
https://github.com/gwxie/Document-Dewarping-with-Control-Points/blob/4c74853b0eb93f7c6006a774c2eb42c64f363531/Source/utilsV4.py#L162
