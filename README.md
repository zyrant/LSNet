# LSNet
This project provides the code and results for 'LSNet: Lightweight Spatial Boosting Network for Detecting Salient Objects in RGB-Thermal Images', IEEE TIP, 2023. [IEEE link](https://ieeexplore.ieee.org/document/10042233)  <br>

# Requirements
Python 3.7+, Pytorch 1.5.0+, Cuda 10.2+, TensorboardX 2.1, opencv-python
if If anything goes wrong with the environment, please check requirements.txt for details.

# Architecture and Details
   ![image](https://user-images.githubusercontent.com/38373305/218299592-13bb523b-8f1d-485f-9c65-137dca4e1544.png)
<img src="https://user-images.githubusercontent.com/38373305/218299628-8b7bbdc5-39b2-4d68-9cdb-828e617c0bab.png" alt="drawing" width="400" height="400"/> <img src="https://user-images.githubusercontent.com/38373305/218299686-8a7e7cae-8970-4e56-a4b1-4986b872741f.png" alt="drawing" width="400" height="400"/>

# Results
<img src="https://user-images.githubusercontent.com/38373305/218301004-4556a1c6-b76b-44b6-aeab-1f48b15cc17d.png" alt="drawing"/>
<img src="https://user-images.githubusercontent.com/38373305/218301024-cbf9bfbc-b3e2-4e44-89a2-106fafeda465.png" alt="drawing"/>
<img src="https://user-images.githubusercontent.com/38373305/218301046-2fab51b0-4566-43d0-a861-9d6ee7136cb1.png" alt="drawing"/>
<img src="https://user-images.githubusercontent.com/38373305/218301207-f40f0a86-247c-4da2-85a2-a9b17fae4ec8.png" alt="drawing" width="1000" height="400"/>

# Dataset and Evaluate tools
Dataset: <br>
RGB-T [baidu](https://pan.baidu.com/s/1fDht3BmqIYPks_iquST5hQ),pin:sf9y / [Google drive](https://drive.google.com/file/d/1vjdD13DTh9mM69mRRRdFBbpWbmj6MSKj/view?usp=share_link) <br>
RGB-D  [baidu](https://pan.baidu.com/s/1A-fwxAtnwMPuznn1PCATWg),pin:7pi5 / [Google drive](https://drive.google.com/file/d/1WzTuHQJCKPE5OreanoU0N2e82Y1_VZyA/view?usp=share_link) <br>
Evaluate tools:<br> 
[matlab](https://github.com/DengPingFan/CODToolbox) verison or [python](https://github.com/lartpang/PySODMetrics) version.

# Saliency Maps
RGB-T [baidu](https://pan.baidu.com/s/1i5GwM0C0OfE5D5VLXlBkVA) pin：fxsk <br>
RGB-D [baidu](https://pan.baidu.com/s/1bAlk753MeeRG0BLMJXAzxQ) pin：6352 <br>

# Pretraining Models
RGB-T [baidu](https://pan.baidu.com/s/1aGP283gNpb3oosvbq4OSDg) pin: wnoa <br>
RGB-D [baidu](https://pan.baidu.com/s/1aGP283gNpb3oosvbq4OSDg) pin: wnoa <br>

# Citation
'''
@ARTICLE{10042233,
  author={Zhou, Wujie and Zhu, Yun and Lei, Jingsheng and Yang, Rongwang and Yu, Lu},
  journal={IEEE Transactions on Image Processing}, 
  title={LSNet: Lightweight Spatial Boosting Network for Detecting Salient Objects in RGB-Thermal Images}, 
  year={2023},
  pages={1-1},
  doi={10.1109/TIP.2023.3242775}}
'''

# Acknowledgement
The implement of this project is based on the code of [BBS-Net](https://github.com/zyjwuyan/BBS-Net) and [Knowledge-Distillation-Zoo](https://github.com/AberHu/Knowledge-Distillation-Zoo).

# Contact
Please drop me an email for further problems or discussion: zzzyylink@gmail.com or wujiezhou@163.com
