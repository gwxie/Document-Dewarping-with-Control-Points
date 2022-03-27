# Document-Dewarping-with-Control-Points
<div align="center">
  <img width="850" src="https://github.com/gwxie/Document-Dewarping-with-Control-Points/blob/main/rectitify_image.jpg">
  
  <p> A simple yet effective approach to rectify distorted document image by estimating control points and reference points. </p>
  <p>The control points and reference points are composed of the same number of vertices and describe the shape of the document in the image before and after rectifying, respectively. The control points are controllable to facilitate interaction or subsequent adjustment. You can flexibly select post-processing methods and the number of vertices according to different application scenarios.</p>
  
</div>

See [“Document Dewarping with Control Points”](https://arxiv.org/pdf/2203.10543.pdf) for more information.

# Quick Start
- Test `python test.py --data_path_test=./your/test/data/path/`

- Train `python train.py --data_path_train=./your/train/data/path/ --data_path_validate=./your/validate/data/path/ --data_path_test=./your/test/data/path/ --batch_size 32 --schema train --parallel 01`
# Requirements
<p>python >=3.7</p>
<p>pytorch</p>
<p>opencv-python</p>
<p>scipy</p>

# Visualization
![image](https://github.com/gwxie/Document-Dewarping-with-Control-Points/blob/main/compare.jpg)


# Dataset
The training dataset can be synthesised using the [scripts](https://github.com/gwxie/Synthesize-Distorted-Image-and-Its-Control-Points).
