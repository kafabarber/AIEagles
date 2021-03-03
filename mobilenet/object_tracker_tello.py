import sys, time, threading
import pathlib

import tensorflow as tf

from djitellopy import Tello    # 드론제어용
#from IPython.display import display

import cv2

from utils import ops as utils_ops
from utils import label_map_util
from tracker import load_model, run_inference_for_single_image, track
#######################################################
# 아래 함수와 drone_controls.py에서 공유하며 사용해야할 글로벌 변수
from global_variables import frameConfig, tracked_bboxes, resultBbox, droneControl, modeConfig, REF_LIST
from drone_control import up_down, yaw, for_back, left_right, intensity_control
import click_handler    # 프로그램 내 클릭 담당
from draw_bbox_mobile import draw_bbox_mobile

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

def Object_tracking(model_name, output_path, show=True, window_name="output", Track_only = []):
    global modeConfig, tracked_bboxes, resultBbox, REF_LIST
    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = 'data/mscoco_complete_label_map.pbtxt'
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    
    #List Generate for DeepSort
    REF_LIST[0] = list(category_index.keys())   # REF_LIST[0]: KEY_LIST
    REF_LIST[1] = list()    # REF_LIST[1]: VAL_LIST
    for key, value in category_index.items():
        REF_LIST[1].append(value['name'])   # REF_LIST[1]: VAL_LIST

    #Set Track_only
    if (len(Track_only) != 0):
        REF_LIST[2] = list(Track_only)  # REF_LIST[2]: Track_only_list
    '''
    reverse_category_index = {}
    for key, value in category_index.items():
        reverse_category_index[value.get('name')] = key
    '''
    '''
    # If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
    PATH_TO_TEST_IMAGES_DIR = pathlib.Path('test_images')
    TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
    #TEST_IMAGE_PATHS
    '''
    #Load an object detection model:
    model_name = 'ssd_mobilenet_v2_2'
    detection_model = load_model(model_name)

    #model input test
    #print(detection_model.signatures['serving_default'].inputs)
    #print(detection_model.signatures['serving_default'].output_dtypes)
    #print(detection_model.signatures['serving_default'].output_shapes)

    times, times_2 = [], []

    # initialize doubleclick event function
    singletrack = click_handler.SingleID()

    # CONNECT TO TELLO
    startCounter = 0    # 반복문 내에서 takeoff 명령어를 한번만 실행하기 위한 변수 (takeoff 후에는 이 변수가 1이 되어야 함)
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
    cv2.namedWindow(window_name)

    me.takeoff()    # 드론에 이륙 명령 전달

    while True:
        try:
            frame_read = me.get_frame_read()
            myFrame = frame_read.frame
            frame = cv2.resize(myFrame, (frameConfig[0], frameConfig[1]))   # frameConfig[0]: frameWidth / [1]: frameHeight
        except Exception as e:
            print(str(e))
            time.sleep(1.0)
            continue
        try:
            original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        except:
            break
        
        t1 = time.time()

        if(modeConfig[2]):  # modeConfig[2]: DetectMode   # True: Detect object, False: Do not detect object (Just show the original frame)
            tracked_bboxes = track(detection_model, original_frame)    # Object Detection 진행

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
                        #print("droneControl:", droneControl)
                        break

                    if len(tracked_bboxes) == i+1:  # 추적 대상을 놓친 경우\
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

        if (not modeConfig[2]) or modeConfig[1]:  # 물체 인식 해제 또는 수동 조작 모드
            # 쓰레드에서 가져온 값을 드론에 전달
            me.yaw_velocity = int(droneControl[0])  # droneControl[0]: yawValue
            me.for_back_velocity = int(droneControl[1] * droneControl[4])   # droneControl[1]: for_backValue / [4]: intensityValue(float)
            me.up_down_velocity = int(droneControl[2] * droneControl[4])    # droneControl[2]: up_downValue
            me.left_right_velocity = int(droneControl[3] * droneControl[4]) # droneControl[3]: left_rightValue

        # SEND VELOCITY VALUES TO TELLO
        if (not modeConfig[2]) or modeConfig[1] or (len(tracked_bboxes) > 0 and singletrack.ClickFlag == 1):  # 물체 인식이 해제되어있거나 수동 조작 모드이거나 자동 추적 모드에서 추적된 개체가 있는 경우
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
        
        #draw detection on frame 
        opencvImage = draw_bbox_mobile(original_frame, tracked_bboxes, category_index, tracking=True)
        t2 = time.time()

        times.append(t2-t1)
        
        times = times[-20:]

        ms = sum(times)/len(times)*1000
        fps = 1000 / ms
        
        opencvImage = cv2.putText(opencvImage, "Time: {:.1f} FPS".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

        if output_path != '': out.write(opencvImage)
        if show:
            cv2.setMouseCallback(window_name, singletrack.mouse_position, param=opencvImage)
            cv2.imshow(window_name, opencvImage)

            key = cv2.waitKey(1) & 0xff
            if key == 27 or key == ord('q'):  # Esc or q
                modeConfig[0] = True # modeConfig[0](=exitFlag)를 True로 만들어 쓰레드 종료
                me.send_rc_control(0, 0, 0, 0)  # 이 코드가 없으면 재이륙 시 드론 통제 불가
                me.streamoff()
                me.land()
                break
        
        if modeConfig[0]:    # 메인 쓰레드에서 modeConfig[0](=exitFlag)를 True로 만들면 쓰레드 종료
            me.streamoff()
            cv2.destroyAllWindows()
            sys.exit(0)
        elif key == 32: # 스페이스 바로 수동, 자동 모드 전환
            me.send_rc_control(0, 0, 0, 0)  # 이 코드가 없으면 모드 변경 직후 드론 통제 불가
            singletrack.track_reset()
            modeConfig[1] = True # 모드 전환 / modeConfig[1]: manualMode (False: Object tracking mode, True: Manual control mode)
        
        if modeConfig[0]:    # 메인 쓰레드에서 modeConfig[0](=exitFlag)를 True로 만들면 쓰레드 종료
            me.streamoff()
            cv2.destroyAllWindows()
            sys.exit(0)

    cv2.destroyAllWindows()

    '''
    #inference image
    for idx ,image_path in enumerate(TEST_IMAGE_PATHS):
        image = track(detection_model, image_path)
        print(type(image))
        opencvImage = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        print(type(opencvImage))
        win_name = 'image' + str(idx)
        cv2.imshow(win_name, opencvImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    '''

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
        model_name = 'ssd_mobilenet_v2_2'
        Object_tracking(model_name, "", True, "output", Track_only=["person"])

    except KeyboardInterrupt:   # 프로그램 실행 중 키보드에서 Ctrl-C를 눌렀을 때 동작
        modeConfig[0] = True    # modeConfig[0]: exitFlag

