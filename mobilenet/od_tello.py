#================================================================
#
#   File name   : object_tracker.py
#   Author      : PyLessons
#   Created date: 2020-09-17
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : code to track detected object from video or webcam
#
#================================================================
import os, sys, time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import threading # For parallel action
import cv2
import numpy as np
import tensorflow as tf
#from yolov3.utils import Load_Yolo_model, image_preprocess, postprocess_boxes, nms, draw_bbox, read_class_names
#from yolov3.configs import *
from djitellopy import Tello    # 드론제어용
from pynput import keyboard # 키 입력 감지용


#from deep_sort import nn_matching
#from deep_sort.detection import Detection
#from deep_sort.tracker import Tracker
#from deep_sort import generate_detections as gdet
#######################################################
# 아래 함수와 drone_controls.py에서 공유하며 사용해야할 글로벌 변수
from global_variables import frameConfig, tracked_bboxes, resultBbox, droneControl, modeConfig
from drone_control import up_down, yaw, for_back, left_right, intensity_control
import click_handler    # 프로그램 내 클릭 담당

def ibicf()
    times, times_2 = [], []

    # initialize doubleclick event function
    singletrack = click_handler.SingleID()

    startCounter = 0    # 반복문 내에서 takeoff 명령어를 한번만 실행하기 위한 변수 (takeoff 후에는 이 변수가 1이 되어야 함)
    # CONNECT TO TELLO
    me = Tello()    # 드론 객체
    me.connect()    # 드론에 연결
    me.for_back_velocity = 0    # 속도 변수 초기화
    me.left_right_velocity = 0
    me.up_down_velocity = 0
    me.yaw_velocity = 0
    me.speed = 0
    ########################
    me.streamoff()
    me.streamon()   # 영상 스트리밍 시작

    # by default VideoCapture returns float instead of int
    fps = 0 #int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, codec, fps, (frameConfig[0], frameConfig[1])) # output_path must be .mp4 # frameConfig[0]: frameWidth / [1]: frameHeight
    cv2.namedWindow('output')
    #cv2.setMouseCallback('output', None)    # match the first argument(string) with cv2.imshow

    NUM_CLASS = read_class_names(CLASSES)
    key_list = list(NUM_CLASS.keys()) 
    val_list = list(NUM_CLASS.values())

    while True:
        ################# FLIGHT
        if startCounter == 0:   # 무한루프 중 최초 1회에 한해 이륙 명령 전달
            me.takeoff()
            startCounter = 1
        frame_read = me.get_frame_read()
        myFrame = frame_read.frame
        frame = cv2.resize(myFrame, (frameConfig[0], frameConfig[1]))   # frameConfig[0]: frameWidth / [1]: frameHeight

        try:
            original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        except:
            break
        
        image_data = image_preprocess(np.copy(original_frame), [input_size, input_size])
        #image_data = tf.expand_dims(image_data, 0)
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        t1 = time.time()
        if YOLO_FRAMEWORK == "tf":
            pred_bbox = Yolo.predict(image_data)
        elif YOLO_FRAMEWORK == "trt":
            batched_input = tf.constant(image_data)
            result = Yolo(batched_input)
            pred_bbox = []
            for key, value in result.items():
                value = value.numpy()
                pred_bbox.append(value)
        
        #t1 = time.time()
        #pred_bbox = Yolo.predict(image_data)
        t2 = time.time()
        

# Main Thread
if __name__ == '__main__':
    e = threading.Event()
    up_down_thread = threading.Thread(target=up_down, args=(e, ))   # 상하 이동 처리를 위한 쓰레드
    yaw_thread = threading.Thread(target=yaw, args=(e, ))   # 좌우 회전 처리를 위한 쓰레드
    for_back_thread = threading.Thread(target=for_back, args=(e, )) # 앞뒤 이동 처리를 위한 쓰레드
    left_right_thread = threading.Thread(target=left_right, args=(e, )) # 좌우 이동 처리를 위한 쓰레드 (수동 운전 모드 전용)
    intensity_control_thread = threading.Thread(target=intensity_control, args=(e, )) # 수동 조작 감도 처리를 위한 쓰레드 (수동 운전 모드 전용)
    
    up_down_thread.daemon = True    # 메인 쓰레드가 종료될 경우 서브 쓰레드도 같이 종료하기 위해 daemon 적용
    yaw_thread.daemon = True
    for_back_thread.daemon = True
    left_right_thread.daemon = True
    intensity_control_thread.daemon = True

    try:
        up_down_thread.start()  # 서브 쓰레드 시작
        yaw_thread.start()
        for_back_thread.start()
        left_right_thread.start()
        intensity_control_thread.start()
        e.set() # 쓰레드 내에서 wait을 적용하기 위해 이벤트 전달

        for thread in threading.enumerate(): 
            print(thread.name)
        #for yolo
        #yolo = Load_Yolo_model()
        #Object_tracking(yolo, "", input_size=YOLO_INPUT_SIZE, show=True, iou_threshold=0.1, rectangle_colors=(255,0,0), Track_only = ["person", "book"])
    
        #for mobilenet detection
            
    except KeyboardInterrupt:   # 프로그램 실행 중 키보드에서 Ctrl-C를 눌렀을 때 동작
        modeConfig[0] = True    # modeConfig[0]: exitFlag
