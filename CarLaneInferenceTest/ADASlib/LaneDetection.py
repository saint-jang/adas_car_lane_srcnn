# Code Maker : Kim Geon Woo
# Start Make Day : 2024-02-14
# Last Make Day : 2024-02-14

from myUltrafastLaneDetector import UltrafastLaneDetector, ModelType
import numpy as np
import cv2

# Lane Detaction Model
class LD_Model:
    def __init__(self, model, modelAdd):
        modelType = None
        if modelAdd == "culane":
            modelType = ModelType.CULANE
        elif modelAdd == "tusimple":
            modelType = ModelType.TUSIMPLE
        self.model = UltrafastLaneDetector(model_path=model, model_type=modelType)
        self.cfgImgH = float(self.model.cfg.img_h)
        self.cfgImgW = float(self.model.cfg.img_w)
        self.laneResults = None
        self.laneDetects = None
        self.laneColors = [(0, 0, 255),
                            (0, 255, 0),
                            (255, 0, 0),
                            (0, 255, 255)]
        self.polyColor = (255, 191, 0)
    
    # Input Frame In Model
    def setFrame(self, frame):
        orgImgH = frame.shape[0]
        orgImgW = frame.shape[1]
        _, results, detects = self.model.detect_lanes(frame)
        self.laneDetects = detects
        for n1, lane in enumerate(results):
            for n2, point in enumerate(lane):
                changeW = int((orgImgW * point[0]) / self.cfgImgW)
                changeH = int((orgImgH * point[1]) / self.cfgImgH)
                results[n1][n2] = [changeW, changeH]
        self.laneResults = results

    # Get Lane Data
    def getData(self):
        return self.laneResults, self.laneDetects
    
    # Draw And Get Frame
    def getDraw(self, orgFrame):
        frame = np.copy(orgFrame)
        # Draw My Driving Road
        if (self.laneDetects[1] and self.laneDetects[2]):
            copyFrame = np.copy(frame)
            cv2.fillPoly(copyFrame, 
                         pts=[np.vstack((self.laneResults[1], np.flipud(self.laneResults[2])))],
                         color=self.polyColor)
            frame = cv2.addWeighted(frame, 0.7, copyFrame, 0.3, 0)
        # Draw Lane Points
        for laneNum, lanePoints in enumerate(self.laneResults):
            for lanePoint in lanePoints:
                cv2.circle(frame,
                           (lanePoint[0], lanePoint[1]),
                           3,
                           (self.laneColors[laneNum]),
                           -1)
        return frame