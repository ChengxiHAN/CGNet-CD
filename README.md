# CGNet-CD:https://chengxihan.github.io/

You can still run CGNet in the open-cd repository. https://github.com/likyoo/open-cd

The Pytorch implementation for::gift::gift::gift:
“[Change Guiding Network: Incorporating Change Prior to Guide Change Detection in Remote Sensing Imagery](https://ieeexplore.ieee.org/document/10234560?denied=),” IEEE J. SEL. TOP. APPL. EARTH OBS. REMOTE SENS., PP. 1–17, 2023, DOI: 10.1109/JSTARS.2023.3310208.
 C. HAN, C. WU, H. GUO, M. HU, J.Li AND H. CHEN, :yum::yum::yum:
 
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/change-guiding-network-incorporating-change/change-detection-on-levir-cd)](https://paperswithcode.com/sota/change-detection-on-levir-cd?p=change-guiding-network-incorporating-change)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/change-guiding-network-incorporating-change/change-detection-on-whu-cd)](https://paperswithcode.com/sota/change-detection-on-whu-cd?p=change-guiding-network-incorporating-change)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/change-guiding-network-incorporating-change/change-detection-on-cdd-dataset-season-1)](https://paperswithcode.com/sota/change-detection-on-cdd-dataset-season-1?p=change-guiding-network-incorporating-change)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/change-guiding-network-incorporating-change/change-detection-on-googlegz-cd)](https://paperswithcode.com/sota/change-detection-on-googlegz-cd?p=change-guiding-network-incorporating-change)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/change-guiding-network-incorporating-change/change-detection-on-levir)](https://paperswithcode.com/sota/change-detection-on-levir?p=change-guiding-network-incorporating-change)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/change-guiding-network-incorporating-change/change-detection-on-sysu-cd)](https://paperswithcode.com/sota/change-detection-on-sysu-cd?p=change-guiding-network-incorporating-change)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/change-guiding-network-incorporating-change/change-detection-on-dsifn-cd)](https://paperswithcode.com/sota/change-detection-on-dsifn-cd?p=change-guiding-network-incorporating-change)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/change-guiding-network-incorporating-change/change-detection-on-s2looking)](https://paperswithcode.com/sota/change-detection-on-s2looking?p=change-guiding-network-incorporating-change)

[2 Sep. 2023] Release the first version of the CGNet
![image-20230415](/picture/CGNet.png)
### Requirement  
```bash
-Pytorch 1.8.0  
-torchvision 0.9.0  
-python 3.8  
-opencv-python  4.5.3.56  
-tensorboardx 2.4  
-Cuda 11.3.1  
-Cudnn 11.3  
```
## Training,Test and Visualization Process   

```bash
python train_CGNet.py --epoch 50 --batchsize 8 --gpu_id '1' --data_name 'WHU' --model_name 'CGNet'

python test.py --gpu_id '1' --data_name 'WHU' --model_name 'CGNet'
```
You can change data_name for different datasets like "LEVIR", "WHU", "SYSU", "S2Looking", "CDD", and "DSIFN".
## Test our trained model result 
You can directly test our model by our provided CGNet weights in  `output/WHU, LEVIR, SYSU, S2Looking, CDD, and DSIFN `. Download in  [Baidu Disk](https://pan.baidu.com/s/18wIxa6UHEQ4lvvXDpEjygw?pwd=2023),pwd:2023 :yum::yum::yum:

And also we provide all test results of our CGNet in the CGNetTestResult!!!! Download in CGNetTestResult or [Baidu Disk](https://pan.baidu.com/s/1owj4sPTGqcCJ7dDiRzBhEg?pwd=2023 ),pwd:2023 :yum::yum::yum:

## Dataset Download   
LEVIR-CD：https://justchenhao.github.io/LEVIR/  , our paper split in [Baidu Disk](https://pan.baidu.com/s/1VVry18KFl2MSWS6_IOlYRA?pwd=2023),pwd:2023 

WHU-CD：http://gpcv.whu.edu.cn/data/building_dataset.html ,our paper split in [Baidu Disk](https://pan.baidu.com/s/1ZLmIyWvHnwyzhyl4xt-GwQ?pwd=2023),pwd:2023

SYSU-CD: Our paper split in [Baidu Disk](https://pan.baidu.com/s/1p0QfogZm4BM0dd1a0LTBBw?pwd=2023),pwd:2023

S2Looking-CD: Our paper split in [Baidu Disk](https://pan.baidu.com/s/1wAXPHhCLJTqPX0pC2RBMsg?pwd=2023),pwd:2023

CDD-CD: Our split in [Baidu Disk](https://pan.baidu.com/s/1cwJ0mEhcrbCWOJn5n-N5Jw?pwd=2023),pwd:2023

DSIFN-CD: Our split in [Baidu Disk]( https://pan.baidu.com/s/1-GD3z_eMoQglSJoi9P-6gw?pwd=2023),pwd:2023

Note: We crop all datasets to a slice of 256×256 before training with it.

## Dataset Path Setting
```
 LEVIR-CD or WHU-CD or SYSU-CD or S2Looking-CD
     |—train  
          |   |—A  
          |   |—B  
          |   |—label  
     |—val  
          |   |—A  
          |   |—B  
          |   |—label  
     |—test  
          |   |—A  
          |   |—B  
          |   |—label
  ```        
 Where A contains images of the first temporal image, B contains images of the second temporal image, and label contains ground truth maps.  
 
![image-20230415](/picture/CGNet-2.png)
![image-20230415](/picture/CGNet-3.png)
![image-20230415](/picture/CGNet-4.png)
![image-20230415](/picture/CGNet-5.png)
![image-20230415](/picture/CGNet-6.png)

Although our proposed method of CGNet does not achieve the effect of SOTA on CDD-CD and DSIFN-CD datasets, we still provide our results here for the convenience of peer comparison experiments.

![image-20230415](/picture/HANet-HCGMNet-CGNet.png)

## Acknowledgments
 
Thanks to all my co-authors [Haonan Guo](https://scholar.google.com/citations?user=HvYxc84AAAAJ&hl=en),[Meiqi Hu](https://meiqihu.github.io/),[Jiepan Li](https://henryjiepanli.github.io/Jiepanli_Henry.github.io/), and [Hongruixuan Chen](https://chrx97.com/). Thanks  for their great work!!  


## Citation 

 If you use this code for your research, please cite our papers.  

```
@ARTICLE{10093022,
  author={Han, Chengxi and Wu, Chen and Guo, Haonan and Hu, Meiqi and Jiepan Li, and Chen, Hongruixuan},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={Change Guiding Network: Incorporating Change Prior to Guide Change Detection in Remote Sensing Imagery}, 
  year={2023},
  volume={},
  number={},
  pages={1-17},
  doi={10.1109/JSTARS.2023.3310208}}
```

## Reference  
[1] C. HAN, C. WU, H. GUO, M. HU, J.Li, AND H. CHEN, 
“[Change Guiding Network: Incorporating Change Prior to Guide Change Detection in Remote Sensing Imagery](https://ieeexplore.ieee.org/document/10234560?denied=),” IEEE J. SEL. TOP. APPL.EARTH OBS. REMOTE SENS., PP. 1–17, 2023, DOI:10.1109/JSTARS.2023.3310208 .

[2] C. HAN, C. WU, H. GUO, M. HU, AND H. CHEN, 
“[HANet: A hierarchical attention network for change detection with bi-temporal very-high-resolution remote sensing images](https://ieeexplore.ieee.org/abstract/document/10093022),” IEEE J. SEL. TOP. APPL.EARTH OBS. REMOTE SENS., PP. 1–17, 2023, DOI: 10.1109/JSTARS.2023.3264802.


[3] [HCGMNET: A Hierarchical Change Guiding Map Network For Change Detection](https://doi.org/10.48550/arXiv.2302.10420).

[4]C. Wu et al., "[Traffic Density Reduction Caused by City Lockdowns Across the World During the COVID-19 Epidemic: From the View of High-Resolution Remote Sensing Imagery](https://ieeexplore.ieee.org/abstract/document/9427164)," in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 14, pp. 5180-5193, 2021, doi: 10.1109/JSTARS.2021.3078611.
