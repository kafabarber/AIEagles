import os, sys, time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import threading # For parallel action
import cv2
import numpy as np
import tensorflow as tf
from yolov3.utils import Load_Yolo_model, image_preprocess, postprocess_boxes, nms, draw_bbox, read_class_names
from yolov3.configs import *
from djitellopy import Tello    # 드론제어용
from pynput import keyboard # 키 입력 감지용


from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet
#######################################################
# 아래 함수와 drone_controls.py에서 공유하며 사용해야할 글로벌 변수
from global_variables import frameConfig, tracked_bboxes, resultBbox, droneControl, modeConfig
from drone_control import up_down, yaw, for_back, left_right, intensity_control
import click_handler    # 프로그램 내 클릭 담당


def Object_tracking(Yolo, output_path, input_size=416, show=False, CLASSES=YOLO_COCO_CLASSES, score_threshold=0.3, iou_threshold=0.45, rectangle_colors='', Track_only = []):
    global modeConfig, tracked_bboxes, resultBbox

    # Definition of the parameters
    max_cosine_distance = 0.7
    nn_budget = None
    
    #initialize deep sort object
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

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

    NUM_CLASS = read_class_names(CLASSES)
    key_list = list(NUM_CLASS.keys()) 
    val_list = list(NUM_CLASS.values())

    AverageFPS = 0
    frameCounter = 0

    while True:
        ################# FLIGHT
        if startCounter == 0:   # 무한루프 중 최초 1회에 한해 이륙 명령 전달
            me.takeoff()
            startCounter = 1
        frame_read = me.get_frame_read()
        myFrame = frame_read.frame
        frame = cv2.resize(myFrame, (frameConfig[0], frameConfig[1]))   # frameConfig[0]: frameWidth / [1]: frameHeigh
        frameCounter = frameCounter + 1

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
        
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        bboxes = postprocess_boxes(pred_bbox, original_frame, input_size, score_threshold)
        bboxes = nms(bboxes, iou_threshold, method='nms')

        # extract bboxes to boxes (x, y, width, height), scores and names
        boxes, scores, names = [], [], []
        for bbox in bboxes:
            if len(Track_only) !=0 and NUM_CLASS[int(bbox[5])] in Track_only or len(Track_only) == 0:
                boxes.append([bbox[0].astype(int), bbox[1].astype(int), bbox[2].astype(int)-bbox[0].astype(int), bbox[3].astype(int)-bbox[1].astype(int)])
                scores.append(bbox[4])
                names.append(NUM_CLASS[int(bbox[5])])

        # Obtain all the detections for the given frame.
        boxes = np.array(boxes) 
        names = np.array(names)
        scores = np.array(scores)
        features = np.array(encoder(original_frame, boxes))
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(boxes, scores, names, features)]

        print("boxes:", boxes)
        
        # Deep_sort only functions in auto tracking mode.
        if True:#modeConfig[1] == True:  # modeConfig[1]: manualMode 
            # Pass detections to the deepsort object and obtain the track information.
            tracker.predict()
            tracker.update(detections)

            # Obtain info from the tracks
            tracked_bboxes = []
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 5:
                    continue 
                bbox = track.to_tlbr() # Get the corrected/predicted bounding box
                class_name = track.get_class() #Get the class name of particular object
                tracking_id = track.track_id # Get the ID for the particular track
                index = key_list[val_list.index(class_name)] # Get predicted object index by object name
                is_highlighted = False  # Define highlighted box: If ture, the box is highlighted (by 이준범)
                tracked_bboxes.append(bbox.tolist() + [tracking_id, index, is_highlighted]) # Structure data, that we could use it with our draw_bbox function
            print("tracked_bboxes:", tracked_bboxes)
        '''
        else:
            # Obtain info from the boxes
            tracked_bboxes = []
            for box in boxes:
                tracked_bboxes.append(box + [-1, 1, False])
            print("tracked_bboxes:", tracked_bboxes)
        '''

        if (singletrack.ClickFlag == 1) and len(tracked_bboxes) > 0:    # 마우스 클릭 신호를 받은 경우
            modeConfig[1] = False  # manualMode = Fasle
            if(singletrack.trackID == None):
                singletrack.MPtoID(tracked_bboxes)
            
            for i in range(len(tracked_bboxes)):
                if tracked_bboxes[i][4] == singletrack.trackID:
                    tracked_bboxes[i][6] = True # is_highlighted = True
                    resultBbox[0] = int(tracked_bboxes[i][2] + tracked_bboxes[i][0]) / 2    # 추적 결과 박스의 중심 x좌표
                    resultBbox[1] = int(tracked_bboxes[i][3] + tracked_bboxes[i][1]) / 2    # 추적 결과 박스의 중심 y좌표
                    resultBbox[2] = int(tracked_bboxes[i][2] - tracked_bboxes[i][0])        # 추적 결과 박스의 폭
                    resultBbox[3] = int(tracked_bboxes[i][3] - tracked_bboxes[i][1])        # 추적 결과 박스의 높이
                    
                    # 쓰레드에서 가져온 값을 드론에 전달
                    me.yaw_velocity = int(droneControl[0])  # droneControl[0]: yawValue
                    me.for_back_velocity = int(droneControl[1]) # droneControl[1]: for_backValue
                    me.up_down_velocity = int(droneControl[2])  # droneControl[2]: up_downValue
                    me.left_right_velocity = int(droneControl[3])   # droneControl[3]: left_rightValue
                    break

                if len(tracked_bboxes) == i+1:  # 추적 대상을 놓친 경우
                    me.send_rc_control(0, 0, 0, 0)  # 이 코드가 없으면 모드 변경 직후 드론 통제 불가\
                    droneControl[0] = 0 # droneControl 리스트 리셋
                    droneControl[1] = 0
                    droneControl[2] = 0
                    droneControl[3] = 0
                    droneControl[4] = 1.0
                    singletrack.track_reset()
                    modeConfig[1] = True # 수동 모드로 전환 / modeConfig[1]: manualMode (False: Object tracking mode, True: Manual control mode)\

        elif not modeConfig[1] and len(tracked_bboxes) == 0:    # 자동 추적 모드에서 추적 대상을 놓친 데 더해 아예 아무것도 감지를 못한 경우
            me.send_rc_control(0, 0, 0, 0)  # 이 코드가 없으면 모드 변경 직후 드론 통제 불가
            droneControl[0] = 0 # droneControl 리스트 리셋
            droneControl[1] = 0
            droneControl[2] = 0
            droneControl[3] = 0
            droneControl[4] = 1.0
            singletrack.track_reset()
            modeConfig[1] = True # 수동 모드로 전환 / modeConfig[1]: manualMode (False: Object tracking mode, True: Manual control mode)

        elif modeConfig[1]:  # 수동 조작 모드
            # 쓰레드에서 가져온 값을 드론에 전달
            me.yaw_velocity = int(droneControl[0])  # droneControl[0]: yawValue
            me.for_back_velocity = int(droneControl[1] * droneControl[4])   # droneControl[1]: for_backValue / [4]: intensityValue(float)
            me.up_down_velocity = int(droneControl[2] * droneControl[4])    # droneControl[2]: up_downValue
            me.left_right_velocity = int(droneControl[3] * droneControl[4]) # droneControl[3]: left_rightValue

        # SEND VELOCITY VALUES TO TELLO
        if (len(tracked_bboxes) > 0 and singletrack.ClickFlag == 1) or modeConfig[1]:  # 수동 조작 모드이거나 자동 추적 모드에서 추적된 개체가 있는 경우
            me.send_rc_control(
                me.left_right_velocity, me.for_back_velocity, me.up_down_velocity, me.yaw_velocity)
        else:   # 추적된 개체가 없으면 가만히 있기
            me.send_rc_control(0, 0, 0, 0)
            droneControl[0] = 0 # droneControl 리스트 리셋
            droneControl[1] = 0
            droneControl[2] = 0
            droneControl[3] = 0
            droneControl[4] = 1.0
            print("Nowhere to go!")
        print("manualMode:", modeConfig[1])

        # draw detection on frame
        image = draw_bbox(original_frame, tracked_bboxes, CLASSES=CLASSES, tracking=True)
        ##print(tracked_bboxes)

        t3 = time.time()
        times.append(t2-t1)
        times_2.append(t3-t1)
        
        times = times[-20:]
        times_2 = times_2[-20:]

        ms = sum(times)/len(times)*1000
        fps = 1000 / ms
        fps2 = 1000 / (sum(times_2)/len(times_2)*1000)

        AverageFPS = (AverageFPS * frameCounter + fps2) / (frameCounter + 1)
        
        image = cv2.putText(image, "Time: {:.1f} FPS".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

        # draw original yolo detection
        #image = draw_bbox(image, bboxes, CLASSES=CLASSES, show_label=False, rectangle_colors=rectangle_colors, tracking=True)

        #print("Time: {:.2f}ms, Detection FPS: {:.1f}, total FPS: {:.1f}".format(ms, fps, fps2))
        if output_path != '': out.write(image)
        if show:
            cv2.setMouseCallback('output', singletrack.mouse_position, param=image)
            cv2.imshow('output', image)
            
            key = cv2.waitKey(1) & 0xff
            if key == 27 or key == ord('q'):  # Esc or q
                modeConfig[0] = True # modeConfig[0](=exitFlag)를 True로 만들어 쓰레드 종료
                me.send_rc_control(0, 0, 0, 0)  # 이 코드가 없으면 재이륙 시 드론 통제 불가
                me.streamoff()
                me.land()
                break
            elif key == 32: # 스페이스 바로 수동, 자동 모드 전환
                me.send_rc_control(0, 0, 0, 0)  # 이 코드가 없으면 모드 변경 직후 드론 통제 불가
                singletrack.track_reset()
                modeConfig[1] = True # 모드 전환 / modeConfig[1]: manualMode (False: Object tracking mode, True: Manual control mode)
        
        if modeConfig[0]:    # 메인 쓰레드에서 modeConfig[0](=exitFlag)를 True로 만들면 쓰레드 종료
            me.streamoff()
            cv2.destroyAllWindows()
            print("AverageFPS:", AverageFPS)
            sys.exit(0)
    
    print("AverageFPS:", AverageFPS)
    cv2.destroyAllWindows()
    print("[DEBUG] main_thread exit.")

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

        yolo = Load_Yolo_model()
        Object_tracking(yolo, "", input_size=YOLO_INPUT_SIZE, show=True, iou_threshold=0.1, rectangle_colors=(255,0,0), Track_only = ["person", "book"])
    except KeyboardInterrupt:   # 프로그램 실행 중 키보드에서 Ctrl-C를 눌렀을 때 동작
        modeConfig[0] = True    # modeConfig[0]: exitFlag
