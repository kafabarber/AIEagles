# AIEagles
Control a drone with hand motion or keyboard / Make a drone track selected object automatically with YOLO and mobilenet.   
NOTE: Custom models in this repository only contains trained data for soldiers and fighter planes. People with casual cloth and other aircrafts cannot be tracked with this model. Use general pretrained model instead.

# General Preperation
1. Make sure your system has proper NVIDIA graphic driver installed. (nouveau driver is discouraged)
2. Install CUDA 10.2
3. Install nvidia-cuda-toolkit
```
sudo apt install nvidia-cuda-toolkit
```
4. Install pip3
```
sudo apt install python3-pip
```
5. Install tensorflow (>= 2.3.1)
```
pip3 install tensorflow tensorflow-gpu
```
6. Install OpenCV (>= 4.1.2)   
Versions above 4.x may be fine
```
pip3 install opencv-python
```
7. Install libraries for controling drone.
```
pip3 install pynput djitellopy
```
8. Install other required python3 libraries.
```
sudo apt install protobuf-compiler
pip3 install tf_slim object_detection scipy 
```

# HandMotion

### Original source code
We used source code from 'AlexeyAB/darknet'(which was also forked from pjreddie/darknet) as base for darknet.   
This code is a fork from the repository below:   
https://github.com/AlexeyAB/darknet   

### Execution
1. Connect your webcam to your computer. Check your camera index. If it is not 0, then edit camera index value to yours inside the source code.
2. Connect a Tello drone.
3. Execute main.py.

# YOLO

### Original source code
We used source code from 'pythonlessons' as base for YOLO object detection.   
This YOLO code is a fork from the repository below:   
https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3   
※ This repository does not contain source code for object tracking(YOLO) local video files since it is already in the repository above.
## Installation
1. Download checkpoints and model_data directory and merge.   
https://afacloud.ml/index.php/s/5nQFwpZmEriCxtQ   
2. Open YOLO/yolov3/configs.py and edit "YOLO_CUSTOM_WEIGHTS" to True(for soldier and fighter plane tracking) or False(for general tracking).   
3. Make sure CUDA is working, and run object_tracker_tello.py
```
python3 object_tracker_tello.py
```


# MobileNet

### Original source code
We used source code from TensorFlow Model Garden as base for Mobilenet object detection.

## Installation
1. Clone TensorFlow Model repository anywhere you want.
```
git clone https://github.com/tensorflow/models.git
```
2. Inside the repository you just cloned, navigate to: models/research/
3. Open terminal from the directory and paste the code below.
Replace [/PATH/TO/MODELS/] to your actual path!
```
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:[/PATH/TO/MODELS/]research/:[/PATH/TO/MODELS/]research/slim
```
4. Execute object_tracker_video.py to track objects in your video file.
5. Connect your Tello drone and execute object_tracker_tello.py to track objects in the real life.
