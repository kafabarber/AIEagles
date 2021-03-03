import sys, time, threading
import pathlib

import tensorflow as tf

import cv2

from utils import ops as utils_ops
from utils import label_map_util
from tracker import load_model, run_inference_for_single_image, track
#######################################################
# 아래 함수와 drone_controls.py에서 공유하며 사용해야할 글로벌 변수
from global_variables import frameConfig, tracked_bboxes, modeConfig, REF_LIST
from draw_bbox_mobile import draw_bbox_mobile

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

video_path   = "/home/lee/Workspace/Soldier_walking_2.mp4"

def Object_tracking(model_name, video_path, output_path, show=True, window_name="output", Track_only = []):
    global frameConfig, modeConfig, tracked_bboxes, REF_LIST
    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = 'data/label_map.pbtxt'
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
    detection_model = load_model(model_name)

    #model input test
    #print(detection_model.signatures['serving_default'].inputs)
    #print(detection_model.signatures['serving_default'].output_dtypes)
    #print(detection_model.signatures['serving_default'].output_shapes)

    times, times_2 = [], []

    if video_path:
        vid = cv2.VideoCapture(video_path) # detect on video
    else:
        vid = cv2.VideoCapture(0) # detect from webcam

    # by default VideoCapture returns float instead of int
    frameConfig[0] = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)) # frameConfig[0]: frameWidth / [1]: frameHeight
    frameConfig[1] = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, codec, fps, (frameConfig[0], frameConfig[1])) # output_path must be .mp4 # frameConfig[0]: frameWidth / [1]: frameHeight
    cv2.namedWindow(window_name)

    AverageFPS = 0
    frameCounter = 0

    while True:
        _, frame = vid.read()

        try:
            original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        except:
            break
        
        t1 = time.time()
        tracked_bboxes = track(detection_model, original_frame)    # Object Detection 진행
        
        #draw detection on frame 
        opencvImage = draw_bbox_mobile(original_frame, tracked_bboxes, category_index, Text_colors=(220,220,220), rectangle_colors=(0,20,40), tracking=True)
        t2 = time.time()

        times.append(t2-t1)
        
        times = times[-20:]

        ms = sum(times)/len(times)*1000
        fps = 1000 / ms

        if True:
            AverageFPS = (AverageFPS * frameCounter + fps) / (frameCounter + 1)
            frameCounter = frameCounter + 1
        
        opencvImage = cv2.putText(opencvImage, "MobileNet v2", (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        #opencvImage = cv2.putText(opencvImage, "Time: {:.1f} FPS".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

        if output_path != '': out.write(opencvImage)
        if show:
            #cv2.setMouseCallback(window_name, singletrack.mouse_position, param=image)
            cv2.imshow(window_name, opencvImage)

            key = cv2.waitKey(1) & 0xff
            if key == 27 or key == ord('q'):  # Esc or q
                modeConfig[0] = True # modeConfig[0](=exitFlag)를 True로 만들어 쓰레드 종료
                break
        
        if modeConfig[0]:    # 메인 쓰레드에서 modeConfig[0](=exitFlag)를 True로 만들면 쓰레드 종료
            cv2.destroyAllWindows()
            print("AverageFPS:", AverageFPS)
            sys.exit(0)

    print("AverageFPS:", AverageFPS)
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
    model_name = 'my_mobileNet_model_son'
    Object_tracking(model_name, video_path, "Soldier_walking_2_mobilenet.mp4", False, "output", Track_only=[])


