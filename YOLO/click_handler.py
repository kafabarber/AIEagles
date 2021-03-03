import cv2

class SingleID:
    '''
    Tello_drone tracker click detection module.
    @auther 손동희
    '''
    
    ClickFlag = None 
    trackID = None
    #MouseX, MouseY

    def __init__(self):
        self.ClickFlag = 0
        self.mouseX = None
        self.mouseY = None
        self.trackID = None
        print("Single ID Module Loaded!!")

    #Mouse click position function
    def mouse_position(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(param,(x,y),100,(255,0,0),-1)
            self.mouseX, self.mouseY = x,y
            self.ClickFlag = 1

    #Get bbox ID from bboxes by Mouse Click position
    def MPtoID(self, tracked_bboxes):
        for i in range(len(tracked_bboxes)):
            if((self.mouseX>=tracked_bboxes[i][0] and self.mouseX <= tracked_bboxes[i][2]) and (self.mouseY >= tracked_bboxes[i][1] and self.mouseY <= tracked_bboxes[i][3])):
                self.trackID = tracked_bboxes[i][4]
                return tracked_bboxes[i][4]
        
        self.mouseX = None
        self.mouseY = None
        self.trackID = None
        self.clickFlag = 0
        
        print("Chose Wrong Bbox")

    def track_reset(self):
        self.ClickFlag = 0
        self.mouseX = None
        self.mouseY = None
        self.trackID = None