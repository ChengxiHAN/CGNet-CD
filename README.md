# CGNet-CD:https://chengxihan.github.io/
The Pytorch implementation for::gift::gift::gift:
“[Change Guiding Network: Incorporating Change Prior to Guide Change Detection in Remote Sensing Imagery](https://ieeexplore.ieee.org/document/10234560?denied=),” IEEE J. SEL. TOP. APPL. EARTH OBS. REMOTE SENS., PP. 1–17, 2023, DOI: 10.1109/JSTARS.2023.3310208.
 C. HAN, C. WU, H. GUO, M. HU, J.Li AND H. CHEN, :yum::yum::yum:

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
## Traing,Test and Visualization Process   

```bash
python train_CGNet.py --epoch 50 --batchsize 8 --gpu_id '1' --data_name 'WHU' --model_name 'CGNet'
python test.py --gpu_id '1' --data_name 'WHU' --model_name 'CGNet'
```
## Test our trained model result 
You can directly test our model by our provided training weights in  `output/WHU, LEVIR, SYSU, and S2Looking `.

## Dataset Download   
LEVIR-CD：https://justchenhao.github.io/LEVIR/  
 
WHU-CD：http://gpcv.whu.edu.cn/data/building_dataset.html ,our paper split in [Baidu Disk](https://pan.baidu.com/s/16g3H1UsDMgqmXaVjiE319Q?pwd=6969),pwd:6969

SYSU-CD:

S2Looking-CD:

Note: Please crop the LEVIR dataset to a slice of 256×256 before training with it.

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

