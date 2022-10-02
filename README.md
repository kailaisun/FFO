# fusion_framework
Code for A fusion framework for vision-based indoor occupancy estimation

## Dependencies
- The code is tested on Ubuntu 20.04.2,python 3.8,cuda 10.1.

- install torch version 

  ```bash
  pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
  ```



## Installation
 1. Install pytorch

 2. Clone this repository
  ```bash
  git clone https://github.com/kailaisun/FFO
  ```
 3. Install 
  ```bash
  pip install -r requirements.txt
  ```
  

## test
### SCM 

```Bash
python people_detect.py --path <video_path>
```
- result of SCM
![image](https://raw.githubusercontent.com/kailaisun/FFO/main/gif/1.gif)
- You can modify hyperparameters of JointDet module in person_detect.py.
```python 
      result_info = joint_de(head_info, other_info,thresh=0.8,conf=0.6,thresh1=0.8)  #line 50
```
### LCM
- peopeo_count.py conducts LCM (YOLOX+Deepsort) of indoor view.
- python joint.py after you obtained the sequences of two-vision LCM.
- Note that in overhead entrance counting method our video frame rate is downsampled to one-fifth of the original video.
- result of YOLOX+Deepsort
![image](https://raw.githubusercontent.com/kailaisun/FFO/main/gif/2.gif) 
    label of indoor view LCM
    ```
    frame: i, in/out num: y
    frame: k, in/out num: y
    .
    .
    .
    ```
    label of overhead view LCM
    ```
    frame: i, num: y
    frame: i+1, num: y
    frame: i+2, num: y
    .
    .
    .
    ```

### DBF

```bash 
pytho main.py
```

