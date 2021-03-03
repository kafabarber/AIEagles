import sys, os, time
import cv2
import xml.etree.ElementTree as ET

test = False

# 프로그램 실행 시 전달된 인자의 유효성 검사
if len(sys.argv) != 3:  # 이미지를 추출할 동영상이 지정되지 않았거나 너무 많이 지정된 경우
    print("사용법: python3 evaluateIOU.py [정답 디렉토리명] [테스트 디렉토리명]")
    exit()

answerDIR = sys.argv[1]
testDIR = sys.argv[2] # 인자로 전달된 디렉토리명

setInfo = open(testDIR+"/setInfo.txt", 'r')

iouSUM = 0.0
imgCount = 0

while True:
    # Get tracked data from setInfo.txt
    line = setInfo.readline()
    if not line:
        break
    imgCount = imgCount + 1

    parsedLine = line.split(',')
    x1 = int(parsedLine[1])
    y1 = int(parsedLine[4])
    x2 = int(parsedLine[3])
    y2 = int(parsedLine[2])

    # Get answer from xml
    xml_file = answerDIR+'/'+parsedLine[0]
    xml_file, _ = os.path.splitext(xml_file)
    xml_file = xml_file + '.xml'
    tree=ET.parse(open(xml_file))
    root = tree.getroot()
    xmlbox = root.find('object').find('bndbox')
    answerX1 = int(float(xmlbox.find('xmin').text))
    answerY1 = int(float(xmlbox.find('ymin').text))
    answerX2 = int(float(xmlbox.find('xmax').text))
    answerY2 = int(float(xmlbox.find('ymax').text))

    # Evaluate IOU
    # COORDINATES OF THE INTERSECTION BOX
    overlapX1 = max(x1, answerX1)
    overlapY1 = max(y1, answerY1)
    overlapX2 = min(x2, answerX2)
    overlapY2 = min(y2, answerY2)

    # AREA OF OVERLAP - Area where the boxes intersect
    width = (overlapX2 - overlapX1)
    height = (overlapY2 - overlapY1)
    iou = 0.0
    # handle case where there is NO overlap
    if (width<0) or (height <0):
        print("image:", parsedLine[0], "\tx1:", parsedLine[1], "\ty1:", parsedLine[2], "\tx2:", parsedLine[3], "\ty2:", parsedLine[4])
        iou = 0.0
    else:
        area_overlap = width * height

        # COMBINED AREA
        area_a = (x2 - x1) * (y2 - y1)
        area_b = (answerX2 - answerX1) * (answerY2 - answerY1)
        area_combined = area_a + area_b - area_overlap

        # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
        iou = float(area_overlap) / float(area_combined)

    iouSUM = iouSUM + iou

    if(test):   # 좌표가 제대로 들어가 있는지 테스트하는 코드
        image = cv2.imread(testDIR+"/"+parsedLine[0], cv2.IMREAD_ANYCOLOR)
        #cv2.rectangle(image, (answerX1, answerY1), (answerX2, answerY2), (0,180,180), 2)
        cv2.rectangle(image, (int(parsedLine[1]), int(parsedLine[2])), (int(parsedLine[3]), int(parsedLine[4])), (0,180,180), 2)

        cv2.imshow("image", image)
        cv2.waitKey(1)
        time.sleep(0.2)

print("Average IOU:", iouSUM/imgCount)

setInfo.close()