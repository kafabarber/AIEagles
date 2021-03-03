from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet
from global_variables import frameConfig, REF_LIST
import numpy as np

class Pipe():
    #######deep Sort##############
    # Definition of the parameters
    max_cosine_distance = 0.7
    nn_budget = None
    
    #initialize deep sort object
    model_filename = 'model/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
"""
    def dict_to_list(self, output_dict):
        global frameConfig, REF_LIST
        boxes, names, scores = [], [], []
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
            #tracking_id = -1 # 아직 구현 안됨
            index = output_dict['detection_classes'][i]     # Get predicted object index by object name
            #is_highlighted = False  # Define highlighted box: If ture, the box is highlighted (by 이준범)
            '''
            Structure of output_dict['detection_boxes']:
            [0: Y-coordinate of the top-left corner,
            1: X-coordinate of the top-left corner,
            2: Y-coordinate of the bottom-right corner,
            3: X-coordinate of the bottom-right corner]
            ※ Coordinate grows from top-left(0,0) bottom-right(1,1) / range: 0~1 (float)
            ※ [IMPORTANT] Please note that coordinates in this mobilenet code are different from coordinates from yolo code.
            '''
            # mobilenet 추적 결과 나오는 좌표체계와 YOLO 추적 결과 나오는 좌표체계가 다르기 때문에 그에 맞추어 변환함Deep SORT input = (minx,miny,w,h)
            boxes.append([int(output_dict['detection_boxes'][i][1] * frameConfig[0]),
                            int((1-output_dict['detection_boxes'][i][2]) * frameConfig[1]),
                           int( output_dict['detection_boxes'][i][3] * frameConfig[0] - output_dict['detection_boxes'][i][1] * frameConfig[0]),
                           int((1-output_dict['detection_boxes'][i][0]) * frameConfig[1] - (1-output_dict['detection_boxes'][i][2]) * frameConfig[1])
                        ])
            names.append(REF_LIST[0][index-1]) # REF_LIST[0]: KEY_LIST
            scores.append(output_dict['detection_scores'][i])

        return boxes, names, scores

    #mobilenet api는 클래스가 pbtxt로 category_index dict파일로 되어있어서 이에 맞춰 바꿔줌
    #여기서는 dict의 value 값으로 key를 찾아야해서 key와 value가 역으로 배치된 reverse_category_index를 사용함
    #사족: 근데 걍 카테고리인덱스를 list로 바꿨으면 굳이 이거 안해도 됐을듯...?
    def deep_sort_pipe(self, output_dict, original_frame, is_highlighted = False):
        
        tracker = self.tracker
        encoder = self.encoder

        # Obtain all the detections for the given frame.
        boxes, names, scores = self.dict_to_list(output_dict)
        boxes = np.array(boxes) 
        names = np.array(names)
        scores = np.array(scores)
        print(boxes.shape)
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
            index = REF_LIST[0].index(class_name) # Get predicted object index by object name  # REF_LIST[0]: KEY_LIST
            #detections[i].get_confidence() # Get predicted object confidence
            tracked_bboxes.append(bbox.tolist() + [tracking_id, index, is_highlighted, 0.0]) # Structure data, that we could use it with our draw_bbox function

        return tracked_bboxes
"""