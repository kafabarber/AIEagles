from ctypes import *
from djitellopy  import Tello
import math
import random
import cv2
import numpy as np
import os
import time
import threading

def c_array(ctype, values):
	arr = (ctype*len(values))()
	arr[:] = values
	return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int),
                ("uc", POINTER(c_float)),
                ("points", c_int),
                ("embeddings", POINTER(c_float)),
                ("embedding_size", c_int),
                ("sim", c_float),
                ("track_id", c_int)]

class DETNUMPAIR(Structure):
    _fields_ = [("num", c_int),
                ("dets", POINTER(DETECTION))]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]
                
                
lib = CDLL("./libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections 
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)


# add
ndarray_image = lib.ndarray_to_image
ndarray_image.argtypes = [POINTER(c_ubyte), POINTER(c_long), POINTER(c_long)]
ndarray_image.restype = IMAGE

# add
def nparray_to_image(img):
	data = img.ctypes.data_as(POINTER(c_ubyte))
	image = ndarray_image(data, img.ctypes.shape, img.ctypes.strides)
	return image
	
	
def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
	'''
	손의 회전 방향을 딥러닝으로 감지하는 함수
	'''
	im = nparray_to_image(image)
	
	num = c_int(0)
	pnum = pointer(num)
	predict_image(net, im)
	dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)	# 
	num = pnum[0]
	if (nms): do_nms_obj(dets, num, meta.classes, nms);
	
	res = []
	for j in range(num):
		for i in range(meta.classes):
			if dets[j].prob[i] > 0:
				b = dets[j].bbox	# 손을 나타내는 박스
				res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
	res = sorted(res, key=lambda x: -x[1])
	free_image(im)
	free_detections(dets, num)
	
	return res

def get_region(center_x,center_y):

	region=-1

	if ((center_x<=width*2/3)and(center_x>width*1/3)):
		if ((center_y<=height*2/3)and(center_y>height*1/3)):
			region=5
		elif ((center_y<=height)and(center_y>height*2/3)):
			region=8
		elif ((center_y<=height*1/3)and(center_y>0)):
			region=2
		else:
			pass
	elif ((center_x<=width*1/3)and(center_x>0)):
		if ((center_y<=height*2/3)and(center_y>height*1/3)):
			region=4
		elif ((center_y<=height)and(center_y>height*2/3)):
			region=7
		elif ((center_y<=height*1/3)and(center_y>0)):
			region=1
		else:
			pass
	elif ((center_x<=width)and(center_x>height*2/3)):
		if ((center_y<=height*2/3)and(center_y>height*1/3)):
			region=6
		elif ((center_y<=height)and(center_y>height*2/3)):
			region=9
		elif ((center_y<=height*1/3)and(center_y>0)):
			region=3
		else:
			pass
	else:
		pass

	return region

def get_fw(order,boxes):
	'''
	손 박스의 폭을 보고 드론의 앞뒤 이동 판단
	'''

	fw=-1

	box_width=boxes[0][2]

	Normal_width=avg_hand
	L_yaw_width=280
	R_yaw_width=280

	if order==b'Normal':
		if box_width<Normal_width*(0.8):
			fw=0
		elif box_width<Normal_width*(1.1):
			fw=1
		else:
			fw=2
	elif order==b'L_yaw':
		if box_width<L_yaw_width*(0.8):
			fw=0
		elif box_width<L_yaw_width*(1.1):
			fw=1
		else:
			fw=2
	elif order==b'R_yaw':
		if box_width<R_yaw_width*(0.8):
			fw=0
		elif box_width<R_yaw_width*(1.1):
			fw=1
		else:
			fw=2
	else:
		pass

	return fw


# add
def convert_box_value(r):
	'''
	딥러닝을 통해 감지된 손 정보를 튜플 형태로 변환하는 함수
	'''
	boxes = []
	
	for k in range(len(r)):
		width = r[k][2][2]
		height = r[k][2][3]
		center_x = r[k][2][0]
		center_y = r[k][2][1]
		bottomLeft_x = center_x - (width / 2)
		bottomLeft_y = center_y - (height / 2)
		
		x, y, w, h = bottomLeft_x, bottomLeft_y, width, height
		boxes.append((x, y, w, h))
		
	return boxes
	
# add
def draw(image, boxes, order, center_x, center_y):
	for k in range(len(boxes)):
		x, y, w, h = boxes[k]
		region=get_region(center_x,center_y)

		text=str(order)
		text=text[2:-1]

		if text=='Normal':
			draw_color=(204,255,102)
			linelength=avg_hand
		elif text=='L_yaw':
			draw_color=(255,153,0)
			linelength=280
		elif text=='R_yaw':
			draw_color=(51,255,255)
			linelength=280
		else:
			draw_color=(102,102,255)
			linelength=0
		
		top = max(0, np.floor(x + 0.5).astype(int))
		left = max(0, np.floor(y + 0.5).astype(int))
		right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
		bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))
		width13=int(width*(1/3))
		width23=int(width*(2/3))
		height13=int(height*(1/3))
		height23=int(height*(2/3))

		if region==1:
			cv2.rectangle(image,(0,0),(width13,height13),(102,102,102),2)
		elif region==2:
			cv2.rectangle(image,(width13,0),(width23,height13),(102,102,102),2)
		elif region==3:
			cv2.rectangle(image,(width23,0),(int(width),height13),(102,102,102),2)
		elif region==4:
			cv2.rectangle(image,(0,height13),(width13,height23),(102,102,102),2)
		elif region==5:
			cv2.rectangle(image,(width13,height13),(width23,height23),(102,102,102),2)
		elif region==6:
			cv2.rectangle(image,(width23,height13),(int(width),height23),(102,102,102),2)
		elif region==7:
			cv2.rectangle(image,(0,height23),(width13,int(height)),(102,102,102),2)
		elif region==8:
			cv2.rectangle(image,(width13,height23),(width23,int(height)),(102,102,102),2)
		elif region==9:
			cv2.rectangle(image,(width23,height23),(int(width),int(height)),(102,102,102),2)
		else:
			pass
		
		cv2.circle(image, (int(center_x),int(center_y)), 5, (0,0,255), 2)
		cv2.rectangle(image, (top,left), (right, bottom), draw_color, 2)
		cv2.putText(image, text=text, org=(int(x),int(y)), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=2, color=draw_color, thickness=2)
		cv2.line(image,(int(x),int(y+h)+15),(int(x+linelength),int(y+h)+15),draw_color,thickness=2)
		
	
def detect_class(order):
	'''
	딥러닝 판별 후 좌회전인지 우회전인지 판단
	'''
	global flag_class
	global flag_land
	global yaw_velocity
	global land_count

	if order==b'Normal':
		yaw_velocity=0
	elif order==b'L_yaw':
		yaw_velocity=-35
	elif order==b'R_yaw':
		yaw_velocity=35
	else:
		land_count=land_count+1
		if land_count>5:
			tello.land()
			flag_land=True
			return

	flag_class=True

def detect_location(center_x,center_y):
	'''
	손바닥의 중앙 위치를 보고 드론의 이동 판단
	'''
	global up_down_velocity
	global left_right_velocity
	global flag_location
	up_down_velocity=left_right_velocity=0

	region=get_region(center_x,center_y)

	if region==1:
		up_down_velocity=30
		left_right_velocity=-30
	elif region==2:
		up_down_velocity=30
	elif region==3:
		up_down_velocity=30
		left_right_velocity=30
	elif region==4:
		left_right_velocity=-30
	elif region==5:
		pass
	elif region==6:
		left_right_velocity=30
	elif region==7:
		up_down_velocity=-30
		left_right_velocity=-30
	elif region==8:
		up_down_velocity=-30
	elif region==9:
		up_down_velocity=-30
		left_right_velocity=30
	else:
		pass
		
	flag_location=True

def detect_size(order,boxes):
	'''
	박스의 폭을 보고 드론의 전후 이동을 판단하는 함수
	'''
	global flag_size
	global forward_backward_velocity
	forward_backward_velocity=0
	
	fw=get_fw(order,boxes)
	if fw==2:
		forward_backward_velocity=30
	elif fw==0:
		forward_backward_velocity=-30
	else:
		pass
	
	flag_size=True
		
# add
if __name__ == "__main__":
	net = load_net(b"./yolov3.cfg", b"./yolov3_best.weights", 0)
	meta = load_meta(b"./custom.data")

	global forward_backward_velocity
	global left_right_velocity
	global yaw_velocity
	global up_down_velocity
	global land_count	# 손 모양이 land로 인식되더라도 일정 이상 대기하기 위한 카운터
	global avg_hand		# 손 폭 저장하는 변수
	global flag_class	# 
	global flag_location
	global flag_size
	global flag_land
	forward_backward_velocity=left_right_velocity=yaw_velocity=land_count = 0
	flag_class=flag_location=flag_size=flag_land=False
	stop_flag=False

	global count	# 처음 16 프레임 동안 초기 손 크기의 평균 폭을 구하기 위한 카운트 변수
	global total
	count=total=0
	
	tello = Tello()
	tello.connect()	# 드론 연결
	tello.streamon()
	frame_read = tello.get_frame_read()
	tello.takeoff()
	
	cap = cv2.VideoCapture(0)	# 카메라 연결(드론과는 별개의 카메라)
	
	width = cap.get(3)	# 카메라 영상 폭
	height = cap.get(4)	# 카메라 영상 높이
	print(width,height)
	
	while(cap.isOpened()):
	
		img = frame_read.frame	# 드론으로부터 영상을 가져옴
		cv2.imshow("Drone_CAM", img)
		cv2.moveWindow('Drone_CAM',800,50)	# 보기 편하게 드론 영상을 옮김
		
		ret, frame = cap.read()	# 카메라부터 영상을 가져옴
		frame = cv2.flip(frame, 1)	# 사용자가 영상을 보고 거울처럼 느끼게 하기 위해 좌우반전
		
		if not ret:	# 카메라가 없으면 반복문 종료
			break
		
		r = detect(net, meta, frame)	# 카메라에서 손 모양 감지
		if len(r) == 1:	# 손이 감지된 경우
			order = r[0][0]	# 드론의 회전 방향 명령
			
			center_x = r[0][2][0]	# 인식된 손 박스의 중앙 좌표
			center_y = r[0][2][1]
			boxes = convert_box_value(r)	# 감지된 손 정보를 튜플 배열로 변환
			
			if count<16:	# 처음 16 프레임 동안 초기 손 크기의 평균 폭을 구함
				count=count+1
				length=boxes[0][2]
				total=total+length
				avg_hand=total/count
				print('평균 길이 : ', avg_hand)
				continue

			draw(frame, boxes, order, center_x, center_y)	# 박스그리기
			if flag_land==True:
				print('Landing...')
				continue

			if flag_class==False and flag_location==False and flag_size==False:
				th_class=threading.Thread(target=detect_class,args=(order,))
				th_location=threading.Thread(target=detect_location,args=(center_x,center_y))
				th_size=threading.Thread(target=detect_size,args=(order,boxes))

				th_class.start()
				th_location.start()
				th_size.start()
			elif flag_class==True and flag_location==True and flag_size==True:
				tello.send_rc_control(left_right_velocity, forward_backward_velocity, up_down_velocity, yaw_velocity)
				flag_class=flag_location=flag_size=False
			else:
				pass

			stop_flag=True

		else:	# 손이 감지되지 않은 경우
			if stop_flag==True:
				tello.send_rc_control(0,0,0,0)
				stop_flag = False
				
		cv2.imshow('frame', frame)
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break




def network_width(net):
    return lib.network_width(net)


def network_height(net):
    return lib.network_height(net)


def bbox2points(bbox):
    """
    From bounding box yolo format
    to corner points cv2 rectangle
    """
    x, y, w, h = bbox
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def class_colors(names):
    """
    Create a dict with one random BGR color for each
    class name
    """
    return {name: (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)) for name in names}


def load_network(config_file, data_file, weights, batch_size=1):
    """
    load model description and weights from config files
    args:
        config_file (str): path to .cfg model file
        data_file (str): path to .data model file
        weights (str): path to weights
    returns:
        network: trained model
        class_names
        class_colors
    """
    network = load_net_custom(
        config_file.encode("ascii"),
        weights.encode("ascii"), 0, batch_size)
    metadata = load_meta(data_file.encode("ascii"))
    class_names = [metadata.names[i].decode("ascii") for i in range(metadata.classes)]
    colors = class_colors(class_names)
    return network, class_names, colors


def print_detections(detections, coordinates=False):
    print("\nObjects:")
    for label, confidence, bbox in detections:
        x, y, w, h = bbox
        if coordinates:
            print("{}: {}%    (left_x: {:.0f}   top_y:  {:.0f}   width:   {:.0f}   height:  {:.0f})".format(label, confidence, x, y, w, h))
        else:
            print("{}: {}%".format(label, confidence))


def draw_boxes(detections, image, colors):
    import cv2
    for label, confidence, bbox in detections:
        left, top, right, bottom = bbox2points(bbox)
        cv2.rectangle(image, (left, top), (right, bottom), colors[label], 1)
        cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
                    (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colors[label], 2)
    return image


def decode_detection(detections):
    decoded = []
    for label, confidence, bbox in detections:
        confidence = str(round(confidence * 100, 2))
        decoded.append((str(label), confidence, bbox))
    return decoded


def remove_negatives(detections, class_names, num):
    """
    Remove all classes with 0% confidence within the detection
    """
    predictions = []
    for j in range(num):
        for idx, name in enumerate(class_names):
            if detections[j].prob[idx] > 0:
                bbox = detections[j].bbox
                bbox = (bbox.x, bbox.y, bbox.w, bbox.h)
                predictions.append((name, detections[j].prob[idx], (bbox)))
    return predictions


def detect_image(network, class_names, image, thresh=.5, hier_thresh=.5, nms=.45):
    """
        Returns a list with highest confidence class and their bbox
    """
    pnum = pointer(c_int(0))
    predict_image(network, image)
    detections = get_network_boxes(network, image.w, image.h,
                                   thresh, hier_thresh, None, 0, pnum, 0)
    num = pnum[0]
    if nms:
        do_nms_sort(detections, num, len(class_names), nms)
    predictions = remove_negatives(detections, class_names, num)
    predictions = decode_detection(predictions)
    free_detections(detections, num)
    return sorted(predictions, key=lambda x: x[1])


#  lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
#  lib = CDLL("libdarknet.so", RTLD_GLOBAL)
hasGPU = True
if os.name == "nt":
    cwd = os.path.dirname(__file__)
    os.environ['PATH'] = cwd + ';' + os.environ['PATH']
    winGPUdll = os.path.join(cwd, "yolo_cpp_dll.dll")
    winNoGPUdll = os.path.join(cwd, "yolo_cpp_dll_nogpu.dll")
    envKeys = list()
    for k, v in os.environ.items():
        envKeys.append(k)
    try:
        try:
            tmp = os.environ["FORCE_CPU"].lower()
            if tmp in ["1", "true", "yes", "on"]:
                raise ValueError("ForceCPU")
            else:
                print("Flag value {} not forcing CPU mode".format(tmp))
        except KeyError:
            # We never set the flag
            if 'CUDA_VISIBLE_DEVICES' in envKeys:
                if int(os.environ['CUDA_VISIBLE_DEVICES']) < 0:
                    raise ValueError("ForceCPU")
            try:
                global DARKNET_FORCE_CPU
                if DARKNET_FORCE_CPU:
                    raise ValueError("ForceCPU")
            except NameError as cpu_error:
                print(cpu_error)
        if not os.path.exists(winGPUdll):
            raise ValueError("NoDLL")
        lib = CDLL(winGPUdll, RTLD_GLOBAL)
    except (KeyError, ValueError):
        hasGPU = False
        if os.path.exists(winNoGPUdll):
            lib = CDLL(winNoGPUdll, RTLD_GLOBAL)
            print("Notice: CPU-only mode")
        else:
            # Try the other way, in case no_gpu was compile but not renamed
            lib = CDLL(winGPUdll, RTLD_GLOBAL)
            print("Environment variables indicated a CPU run, but we didn't find {}. Trying a GPU run anyway.".format(winNoGPUdll))
else:
    lib = CDLL(os.path.join(
        os.environ.get('DARKNET_PATH', './'),
        "libdarknet.so"), RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

copy_image_from_bytes = lib.copy_image_from_bytes
copy_image_from_bytes.argtypes = [IMAGE,c_char_p]

predict = lib.network_predict_ptr
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

if hasGPU:
    set_gpu = lib.cuda_set_device
    set_gpu.argtypes = [c_int]

init_cpu = lib.init_cpu

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_batch_detections = lib.free_batch_detections
free_batch_detections.argtypes = [POINTER(DETNUMPAIR), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict_ptr
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

load_net_custom = lib.load_network_custom
load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_custom.restype = c_void_p

free_network_ptr = lib.free_network_ptr
free_network_ptr.argtypes = [c_void_p]
free_network_ptr.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

predict_image_letterbox = lib.network_predict_image_letterbox
predict_image_letterbox.argtypes = [c_void_p, IMAGE]
predict_image_letterbox.restype = POINTER(c_float)

network_predict_batch = lib.network_predict_batch
network_predict_batch.argtypes = [c_void_p, IMAGE, c_int, c_int, c_int,
                                   c_float, c_float, POINTER(c_int), c_int, c_int]
network_predict_batch.restype = POINTER(DETNUMPAIR)
