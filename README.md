# FFO
Code for [A fusion framework for vision-based indoor occupancy estimation.](https://doi.org/10.1016/j.buildenv.2022.109631)

## Environment
- The code is tested on Ubuntu 20.04.2, python 3.8, cuda 11.1.


## Installation
 1. Install pytorch

  ```bash
  pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
  ```

 2. Clone this repository
  ```bash
  git clone https://github.com/kailaisun/FFO
  ```
  
 3. Install 
  ```bash
  pip install -r requirements.txt
  ```
  

## Test
### SCM ( Scene-based counting method)

```Bash
python people_detect.py --path <video_path>
```
- Result of SCM


![image](https://raw.githubusercontent.com/kailaisun/FFO/main/gif/1.gif)
- You can modify hyperparameters of JointDet module in person_detect.py.
```python 
 result_info = joint_de(head_info, other_info,thresh=0.8,conf=0.6,thresh1=0.8)  #line 50
```
### LCM ( Line-based counting method)
- peopeo_count.py conducts LCM (YOLOX+Deepsort) of indoor view.
- After you obtained the sequences of two-vision LCM:
```Bash
python joint.py
```
- Note that in overhead entrance counting method our video frame rate is downsampled to one-fifth of the original video.
- Result of YOLOX+Deepsort


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

