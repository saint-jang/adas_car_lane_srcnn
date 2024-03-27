# Code Maker : Kim Geon Woo
# Start Make Day : 2024-02-15
# Last Make Day : 2024-02-16

import math
import cv2
import numpy as np

# input lane data, output lane angle and output action of lane change
class LaneChange:
    def __init__(self):
        self.leftAngle = None
        self.rightAngle = None
        self.Lcolor = (0, 255, 0)
        self.Rcolor = (255, 0, 0)
        self.lineLen = 0.4
        self.Lpoint = None
        self.Rpoint = None
        self.leftDeltaAngle = None
        self.rightDeltaAngle = None
        self.safeDelta = 1.5
        self.safeAngle = 55.0
        self.lSign = None
        self.rSign = None
        self.lVariange = None
        self.rVariance = None
        self.safeVarince = 10.0
        self.LlastLane = []
        self.RlastLane = []

    # input left lane data
    def setLeftLane(self, lane):
        angle, self.lVariange = self.getAngleInLane(lane)
        self.leftDeltaAngle = self.setDeltaAngle(self.leftAngle, angle)
        self.leftAngle = angle
        if len(lane) >= 1:
            self.Lpoint = lane[0]
        self.lSign = self.setWarningSign(self.leftDeltaAngle, self.leftAngle, self.lVariange)
        self.setLastLane(lane, "L")

    # input right lane data
    def setRightLane(self, lane):
        angle, self.rVariance = self.getAngleInLane(lane)
        self.rightDeltaAngle = self.setDeltaAngle(self.rightAngle, angle)
        self.rightAngle = angle
        if len(lane) >= 1:
            self.Rpoint = lane[0]
        self.rSign = self.setWarningSign(self.rightDeltaAngle, self.rightAngle, self.rVariance)
        self.setLastLane(lane, "R")

    # set last lane
    def setLastLane(self, lane, laneName):
        if (laneName == "L") and (self.lSign != 0):
            self.LlastLane = lane
        if (laneName == "R") and (self.rSign != 0):
            self.RlastLane = lane

    # get last lane to LDmodel data
    def getLastLane2LDmodelData(self, results, detects):
        dstResults = results.copy()
        dstDetects = detects.copy()
        vLane = [False, False, False, False]
        if (self.lSign == 0) and (len(self.LlastLane) >= 2):
            dstResults[1] = self.LlastLane
            dstDetects[1] = True
            vLane[1] = True
        if (self.rSign == 0) and (len(self.RlastLane) >= 2):
            dstResults[2] = self.RlastLane
            dstDetects[2] = True
            vLane[2] = True
        return dstResults, dstDetects, vLane


    # input lane, output angle, variance
    def getAngleInLane(self, lane):
        if len(lane) <= 1:
            return None, None
        else:
            p1 = lane[0]
            angleList = []
            for i in range(1, len(lane)):
                p2 = lane[i]
                angleList.append(self.getAngleInPoints(p1, p2))
                p1 = p2
            angle = self.getAverage(angleList)
            return angle, self.getVariance(angleList, angle)
        
    # input list, output average
    def getAverage(self, lt):
        all_ = sum(lt)
        return all_ / len(lt)
    
    # input list, output variance
    def getVariance(self, lt, avg):
        del_list = []
        for ang in lt:
            del_list.append((ang - avg)**2)
        all_ = sum(del_list)
        return all_ / len(lt)
    
    # output variance
    def outputVariance(self):
        return self.lVariange, self.rVariance

    # input point1 and point2, output angle 
    def getAngleInPoints(self, p1, p2):
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]
        xLen = None
        if x1 > x2:
            xLen = x1 - x2
        else:
            xLen = x2 - x1
        yLen = None
        if y1 > y2:
            yLen = y1 - y2
        else:
            yLen = y2 - y1
        return np.degrees(math.atan2(yLen, xLen))

    # output left angle
    def getAngle(self):
        return self.leftAngle, self.rightAngle
    
    # draw lane in frame
    def drawLane(self, frame):
        dstFrame = np.copy(frame)
        dstFrame = self.drawLeftLane(dstFrame)
        dstFrame = self.drawRightLane(dstFrame)
        return dstFrame
    
    # draw left lane in frame
    def drawLeftLane(self, frame):
        if self.leftAngle == None:
            return frame
        else:
            dstFrame = np.copy(frame)
            imgH, imgW = dstFrame.shape[:2]
            px, py = self.Lpoint
            # y = mx + b
            radian = np.radians(self.leftAngle)
            p1 = (0, 0)
            lineLen = int(imgH * self.lineLen)
            p2 = (int(lineLen * math.cos(radian)), int(lineLen * math.sin(radian)))
            # move line
            p2 = (p2[0], p2[1] * -1)
            p1 = (p1[0] + px, p1[1] + py)
            p2 = (p2[0] + px, p2[1] + py)
            # draw
            dstFrame = cv2.line(dstFrame, p1, p2, self.Lcolor, 4)
            return dstFrame

    # draw right lane in frame
    def drawRightLane(self, frame):
        if self.rightAngle == None:
            return frame
        else:
            dstFrame = np.copy(frame)
            imgH, imgW = dstFrame.shape[:2]
            px, py = self.Rpoint
            # y = mx + b
            radian = np.radians(self.rightAngle)
            p1 = (0, 0)
            lineLen = int(imgH * self.lineLen)
            p2 = (int(lineLen * math.cos(radian)), int(lineLen * math.sin(radian)))
            # move line
            p2 = (p2[0] * -1, p2[1] * -1)
            p1 = (p1[0] + px, p1[1] + py)
            p2 = (p2[0] + px, p2[1] + py)
            # draw
            dstFrame = cv2.line(dstFrame, p1, p2, self.Rcolor, 4)
            return dstFrame

    # draw angle data in frame
    def drawAngleData(self, frame, textY=10):
        dstFrame = np.copy(frame)
        imgH, imgW = dstFrame.shape[:2]
        if self.leftAngle != None:
            pt = (10, imgH - textY)
            dstFrame = cv2.putText(dstFrame, f"angle : {self.leftAngle:.2f}", pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        if self.rightAngle != None:
            pt = (imgW - 120, imgH - textY)
            dstFrame = cv2.putText(dstFrame, f"angle : {self.rightAngle:.2f}", pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return dstFrame
    
    # draw delta angle data in frame
    def drawDeltaAngleData(self, frame, textY=30):
        dstFrame = np.copy(frame)
        imgH, imgW = dstFrame.shape[:2]
        if self.leftDeltaAngle != None:
            pt = (10, imgH - textY)
            dstFrame = cv2.putText(dstFrame, f"delta : {self.leftDeltaAngle:.2f}", pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        if self.rightDeltaAngle != None:
            pt = (imgW - 120, imgH - textY)
            dstFrame = cv2.putText(dstFrame, f"delta : {self.rightDeltaAngle:.2f}", pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return dstFrame
    
    # draw variance data in frame
    def drawVariance(self, frame, textY=50):
        dstFrame = np.copy(frame)
        imgH, imgW = dstFrame.shape[:2]
        if self.lVariange != None:
            pt = (10, imgH - textY)
            dstFrame = cv2.putText(dstFrame, f"var : {self.lVariange:.2f}", pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        if self.rVariance != None:
            pt = (imgW - 120, imgH - textY)
            dstFrame = cv2.putText(dstFrame, f"var : {self.rVariance:.2f}", pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return dstFrame
    
    # input new angle, output delta angle
    def setDeltaAngle(self, a1, a2):
        if (a1 != None) and (a2 != None):
            return a1 - a2
        else:
            return None

    # output delta angle
    def getDeltaAngle(self):
        return self.leftDeltaAngle, self.rightDeltaAngle
    
    # set warning sign
    def setWarningSign(self, delta, angle, var):
        if (delta == None) or (abs(delta) > self.safeDelta):
            return 0 # no lane
        if (var == None) or (abs(var) > self.safeVarince):
            return 0 # no lane
        if angle <= self.safeAngle:
            return 1 # safe lane
        else:
            return 2 # danger
        
    # draw warning sign
    def drawWarningSign(self, frame, textY = 80):
        dstFrame = np.copy(frame)
        imgH, imgW = dstFrame.shape[:2]
        signs = ["no lane", "safe lane", "danger"]
        colors = [(0, 255, 255), (0, 255, 0), (0, 0, 255)]
        pt = (10, imgH - textY)
        dstFrame = cv2.putText(dstFrame, signs[self.lSign], pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[self.lSign], 2)
        pt = (imgW - 120, imgH - textY)
        dstFrame = cv2.putText(dstFrame, signs[self.rSign], pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[self.rSign], 2)
        return dstFrame
    
    # get warning sign
    def getWarningSign(self):
        signs = ["no lane", "safe lane", "danger"]
        return signs[self.lSign], signs[self.rSign]