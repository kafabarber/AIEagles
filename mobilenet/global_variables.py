'''
object_tracker_tello.py와 drone_control.py에서
사용할 글로벌 변수를 모아놓은 파일

위의 두 파일은 서로 의존적이지만, 제 3의 파일을 만들어
양측에서 모두 접근하여야 하는 변수와 함수를 분리함으로써
Cyclical dependency를 회피하기 위한 목적이다.

이 파일에서 값을 가져올 때는 반드시 Call-by-Reference를 사용할 것
Call-by-Value를 사용할 경우 서로 다른 쓰레드에서 값을 공유하지 못한다.
'''

frameConfig = [640, 480, 50]
'''
[Structure of frameConfig]
[0: frameWidth
 1: frameHeight
 2: deadZone]
'''

tracked_bboxes = [] # 추적 객체 히스토그램 저장 변수
'''
[Structure of bbox]
[0: X-coordinate of the down-left corner,
 1: Y-coordinate of the down-left corner,
 2: X-coordinate of the top-right corner,
 3: Y-coordinate of the top-right corner,
 4: Index of object in the certain category (ex: preson 1, person 3, ...),
 5: Detected object's index (Refer model_data/coco/coco.names),
 6: is_highlighted box (True: highlighted, False: normal)
 7: Detection score]

tracked_bbox contains multiple bboxes for each tracked object.
'''

resultBbox = [0, 0, 0, 0] # 추적 결과 박스의 좌표, 폭, 높이
'''
[Structure of resultBbox]
[0: cx  # 추적 결과 박스의 중심 x좌표
 1: cy  # 추적 결과 박스의 중심 y좌표
 2: w   # 추적 결과 박스의 폭
 3: h   # 추적 결과 박스의 높이]
'''

droneControl = [0, 0, 0, 0, 1.0]
'''
[Structure of droneControl]
 0: yawValue
 1: for_backValue
 2: up_downValue
 3: left_rightValue
 4: intensityValue
'''

modeConfig = [False, True, False]
'''
[Structure of modeConfig]
 0: exitFlag    # 각 쓰레드는 이 글로벌 변수를 매 루프마다 확인하고, True일 경우 쓰레드 종료
 1: manualMode  # False: Object tracking mode, True: Manual control mode
 2: DetectMode   # True: Detect object, False: Do not detect object (Just show the original frame)
'''

'''
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet
'''

#Track_only_list = None
#Deep Sort 관련 Global variable
REF_LIST = [[],[],[]]
'''
[Structure of REF_LIST]
 0: KEY_LIST
 1: VAL_LIST
 2: Track_only_list
'''

import deep_sort_pipe
pipe = deep_sort_pipe.Pipe()