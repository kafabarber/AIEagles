import os
import pathlib

import numpy as np
import tarfile

import tensorflow as tf
from matplotlib import pyplot as plt
import six

#import DeepSort
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet

# Import from local file
from utils import ops as utils_ops
#from utils import visualization_utils as vis_util
from global_variables import frameConfig, tracked_bboxes, pipe, REF_LIST

#initialize Pipe
#pipe = Pipe()

def load_model(model_name):
    base_path = "model/"
    model_file = model_name + '.tar.gz'
    model_dir = base_path + '/' + model_name
    

    #load tar file from local and extract
    #print(os.path.isdir(model_dir))
    if(os.path.isdir(model_dir) == False):
        tar_file = base_path + '/' + model_file
        tar = tarfile.open(tar_file, "r:gz")
        for tarinfo in tar:
            tar.extract(tarinfo, path=(base_path + '/' + model_name))

    model_dir = pathlib.Path(model_dir)

    model = tf.saved_model.load(str(model_dir))

    return model

def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]

    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and takereverse_category_index, index [0] to remove the batch dimension.
    # We're only interested inconv2d_ the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detconv2d_ection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
    return output_dict

def dict_to_list(output_dict, Track_only = []):
    global frameConfig, REF_LIST
    boxes, names, scores = [], [], []
    for i in range(len(output_dict)):
        if output_dict['detection_scores'][i] < 0.4:    # 추적 결과 score가 50% 미만이면 tracked_bboxes에 추가하지 않음
            break
        '''
        if output_dict['detection_classes'][i] in six.viewkeys(category_index):
         class_name = category_index[output_dict['detection_classes'][i]]['name']
        else:
            class_name = 'N/A'
            class_name = str(class_name)
            print("[DEBUG] class_name:", class_name)
        '''
        
        #tracking_id = -1 # 아직 구현 안됨
        index = output_dict['detection_classes'][i]     # Get predicted object index by object name
        #is_highlighted = False  # Define highlighted box: If ture, the box is highlighted (by 이준범)

        if len(Track_only) !=0 and REF_LIST[1][index-1] in Track_only or len(Track_only) == 0: # REF_LIST[1]: VAL_LIST
            '''
            Structure of output_dict['detection_boxes']:
            [0: Y-coordinate of the top-left corner,
            1: X-coordinate of the top-left corner,
            2: Y-coordinate of the bottom-right corner,
            3: X-coordinate of the bottom-right corner]
            ※ Coordinate grows from top-left(0,0) bottom-right(1,1) / range: 0~1 (float)
            ※ [IMPORTANT] Please note that coordinates in this mobilenet code are different from coordinates from yolo code.
            '''
            # mobilenet 추적 결과 나오는 좌표체계와 YOLO 추적 결과 나오는 좌표체계가 다르기 때문에 그에 맞추어 변환함 Deep SORT input = (minx,miny,w,h)
            boxes.append([int(output_dict['detection_boxes'][i][1] * frameConfig[0]),
                        int((output_dict['detection_boxes'][i][0]) * frameConfig[1]),
                       int( output_dict['detection_boxes'][i][3] * frameConfig[0] - output_dict['detection_boxes'][i][1] * frameConfig[0]),
                       int((1-output_dict['detection_boxes'][i][0]) * frameConfig[1] - (1-output_dict['detection_boxes'][i][2]) * frameConfig[1])
                    ])
            names.append(REF_LIST[0][index-1])
            scores.append(output_dict['detection_scores'][i])

    return boxes, names, scores

    #mobilenet api는 클래스가 pbtxt로 category_index dict파일로 되어있어서 이에 맞춰 바꿔줌
    #여기서는 dict의 value 값으로 key를 찾아야해서 key와 value가 역으로 배치된 reverse_category_index를 사용함
    #사족: 근데 걍 카테고리인덱스를 list로 바꿨으면 굳이 이거 안해도 됐을듯...?
def deep_sort_pipe(output_dict, original_frame, is_highlighted = False):
    global pipe, REF_LIST
    tracker = pipe.tracker
    encoder = pipe.encoder

     # Obtain all the detections for the given frame.
    boxes, names, scores = dict_to_list(output_dict, REF_LIST[2])   # REF_LIST[2]: Track_only_list
    boxes = np.array(boxes) 
    names = np.array(names)
    scores = np.array(scores)
    features = np.array(encoder(original_frame, boxes))
        
    detections = [Detection(bbox, score, name, feature) for bbox, score, name, feature in zip(boxes, scores, names, features)]

    # Pass detections to the deepsort object and obtain the track information.
    tracker.predict()
    tracker.update(detections)

    # Obtain info from the tracks 
    tracked_bboxes = []
    for i, track in enumerate(tracker.tracks):
        if not track.is_confirmed() or track.time_since_update > 5:
            continue 
        bbox = track.to_tlbr() # Get the corrected/predicted bounding box
        class_name = track.get_class() #Get the class name of particular object
        tracking_id = track.track_id # Get the ID for the particular track
        index = REF_LIST[0].index(class_name)+1 # Get predicted object index by object name  # REF_LIST[0]: KEY_LIST
        #detections[i].get_confidence() # Get predicted object confidence
        tracked_bboxes.append(bbox.tolist() + [tracking_id, index, is_highlighted, 0.0]) # Structure data, that we could use it with our draw_bbox function

    return tracked_bboxes

def track(model, frame):
    global tracked_bboxes
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = np.array(frame)
    # Actual detection.
    output_dict = run_inference_for_single_image(model, image_np)
    '''
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=4)
    '''
    tracked_bboxes = [] # 이전 루프의 추적 데이터 리셋
    """ #moved to deep sort pipe
    for i in range(len(output_dict)):
        if output_dict['detection_scores'][i] < 0.5:    # 추적 결과 score가 50% 미만이면 tracked_bboxes에 추가하지 않음
            break
        '''
        if output_dict['detection_classes'][i] in six.viewkeys(category_index):
            class_name = category_index[output_dict['detection_classes'][i]]['name']
        else:
            class_name = 'N/A'
        class_name = str(class_name)
        print("[DEBUG] class_name:", class_name)
        '''
        tracking_id = -1 # 아직 구현 안됨
        index = output_dict['detection_classes'][i]     # Get predicted object index by object name
        is_highlighted = False  # Define highlighted box: If ture, the box is highlighted (by 이준범)
        '''
        Structure of output_dict['detection_boxes']:
        [0: Y-coordinate of the top-left corner,
         1: X-coordinate of the top-left corner,
         2: Y-coordinate of the bottom-right corner,
         3: X-coordinate of the bottom-right corner]
         ※ Coordinate grows from top-left(0,0) bottom-right(1,1) / range: 0~1 (float)
         ※ [IMPORTANT] Please note that coordinates in this mobilenet code are different from coordinates from yolo code.
        '''
        # mobilenet 추적 결과 나오는 좌표체계와 YOLO 추적 결과 나오는 좌표체계가 다르기 때문에 그에 맞추어 변환함
        tracked_bboxes.append([int(output_dict['detection_boxes'][i][1] * frameConfig[0]),
                               int((1-output_dict['detection_boxes'][i][0]) * frameConfig[1]),
                               int(output_dict['detection_boxes'][i][3] * frameConfig[0]),
                               int((1-output_dict['detection_boxes'][i][2]) * frameConfig[1]),
                               tracking_id, index, is_highlighted, output_dict['detection_scores'][i]
        ])
    """
    tracked_bboxes = deep_sort_pipe(output_dict, frame)
    #print("[DEBUG] tracked_boxes:\n", tracked_bboxes)
    return tracked_bboxes