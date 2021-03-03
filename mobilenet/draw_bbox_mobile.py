from collections import Counter
import random
import colorsys
import numpy as np
import cv2
'''
def read_class_names(class_file_name):
    # loads class name from a file
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names
'''

#mobilenet api는 클래스가 pbtxt로 category_index dict파일로 되어있어서 이에 맞춰 바꿔줌
def draw_bbox_mobile(image, tracked_bboxes, category_index, show_label=True, show_confidence = False, Text_colors=(30,20,20), rectangle_colors='', tracking=False):   
    NUM_CLASS = category_index
    num_classes = len((NUM_CLASS))
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    #print("hsv_tuples", hsv_tuples)
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(tracked_bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        id_number = bbox[4]
        class_ind = int(bbox[5])
        color_ind = class_ind - 1
        is_highlighted = bbox[6]
        score = bbox[7]
        bbox_color = rectangle_colors if rectangle_colors != '' else colors[color_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 1000)
        if bbox_thick < 1: bbox_thick = 1
        fontScale = 0.75 * bbox_thick
        (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])

        # put object rectangle
        if is_highlighted:
            cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thick*2)  # 강조 표시된 박스는 두께 두배
        else:
            cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thick*2)

        #yolo에서 score자리가 딥소트 돌리면 track id로 바뀌어 스코어가 안나와서 이부분 수정해줌
        if show_label:
            # get text label 
            score_str = " {:.2f}".format(score) if show_confidence else ""

            if tracking: id_str = " "+str(id_number)

            label = NUM_CLASS.get(class_ind).get('name') + id_str + score_str

            # get text size
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                                  fontScale, thickness=bbox_thick)
            # put filled text rectangle
            cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), bbox_color, thickness=cv2.FILLED)

            # put text above rectangle
            cv2.putText(image, label, (x1, y1-4), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        fontScale, Text_colors, bbox_thick, lineType=cv2.LINE_AA)

    return image