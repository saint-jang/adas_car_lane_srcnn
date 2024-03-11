import cv2 
import datetime
import pygame
import numpy as np
import carla
from ADASlib.VehicleDetection import VD_Model
from ADASlib.LaneDetection import LD_Model
from ADASlib.LaneChange import LaneChange

class ADAS:
    def __init__(self):
        VD_MODEL = 'yolov8n_epoch_200_batch_60_best.pt'
        VD_MODEL_ADD = 'mycoco128.txt'
        self.VDmodel = VD_Model(model=VD_MODEL, modelAdd=VD_MODEL_ADD, DS=True)
        LD_MODEL = "epoch_200_batch_50_loss_best.onnx"
        LD_MODEL_ADD = "culane"
        self.LDmodel = LD_Model(LD_MODEL, LD_MODEL_ADD)
        self.laneChange = LaneChange()
        self.changeNum = 0

        self.VDon = False
        self.LDon = False

        self.accel = False
        self.brake = False
        self.left = False
        self.right = False

        self.AutoOn = False

        self.speed = 0.0
        self.angle = 0.0

    def ChangeModel(self):
        self.changeNum += 1
        if self.changeNum == 1:
            self.VDon = True
            self.LDon = False
            self.AutoOn = False
        elif self.changeNum == 2:
            self.VDon = False
            self.LDon = True
            self.AutoOn = False
        elif self.changeNum == 3:
            self.VDon = True
            self.LDon = True
            self.AutoOn = False
        elif self.changeNum == 4:
            self.VDon = True
            self.LDon = True
            self.AutoOn = True
        else:
            self.changeNum = 0
            self.VDon = False
            self.LDon = False
            self.AutoOn = False
            self.AutoCarInit()

    def VDview(self, inputFrame, drawFrame):
        self.VDmodel.setFrame(inputFrame)
        outFrame = self.VDmodel.getDraw(drawFrame)
        return outFrame

    def LDview(self, inputFrame, drawFrame):
        self.LDmodel.setFrame(inputFrame)
        vLane = None
        laneResults, laneDetects = self.LDmodel.getData()
        self.laneChange.setLeftLane(laneResults[1])
        self.laneChange.setRightLane(laneResults[2])
        self.LDmodel.laneResults, self.LDmodel.laneDetects, vLane = self.laneChange.getLastLane2LDmodelData(laneResults, laneDetects)
        outFrame = self.LDmodel.getDraw(drawFrame, vLane=vLane)
        outFrame = self.laneChange.drawAngleData(outFrame)
        outFrame = self.laneChange.drawDeltaAngleData(outFrame)
        outFrame = self.laneChange.drawVariance(outFrame)
        outFrame = self.laneChange.drawWarningSign(outFrame)
        return outFrame

    def Display(self, display):
        if (self.VDon == False) and (self.LDon == False):
            return 0

        start_time = datetime.datetime.now()
        frame = pygame.surfarray.array3d(display)
        frame = np.rot90(frame)
        frame = np.flipud(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        copyFrame = np.copy(frame)

        if self.LDon == True:
            frame = self.LDview(copyFrame, frame)
            cv2.putText(frame, "LD on", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        if self.VDon == True:
            frame = self.VDview(copyFrame, frame)
            cv2.putText(frame, "VD on", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        if self.AutoOn == True:
            self.AutoCar()
            cv2.putText(frame, "Auto on", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        end_time = datetime.datetime.now()
        work_time = (end_time - start_time).total_seconds()
        fps = 1 / work_time
        cv2.putText(frame, f"FPS : {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = pygame.image.frombuffer(frame.flatten(), frame.shape[1::-1], 'RGB')
        display.blit(frame, (0, 0))

    def AutoCar(self):
        if self.laneChange.lSign == 2: # lSign이 Danger(2)라면
            self.right = True
        else:
            self.right = False
        if self.laneChange.rSign == 2: # rSign이 Danger(2)라면
            self.left = True
        else:
            self.left = False
        if self.speed < 5.0:
            self.accel = True
        else:
            self.accel = False

    def AutoCarInit(self):
        self.accel = False
        self.left = False
        self.right = False
        self.brake = False