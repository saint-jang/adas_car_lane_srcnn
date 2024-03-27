# Code Maker : Kim Geon Woo
# Start Make Day : 2024-02-14
# Last Make Day : 2024-02-14

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np

# Vehicle Detection Model
class VD_Model:
    def __init__(self, model, modelAdd, DS=True, DSage=30):
        coco128 = open(modelAdd, 'r')
        self.classList = coco128.read().split('\n')
        coco128.close()
        self.model = YOLO(model)
        self.results = None
        self.inputDS = None
        self.confidenceThreshold = 0.6
        self.green = (0, 255, 0)
        self.white = (255, 255, 255)
        self.DS = DS
        self.DSage = DSage
        self.tracker = DeepSort(max_age=self.DSage)

    # Set Confidence Threshold
    def setCongidenceThreshold(self, num):
        self.confidenceThreshold = num

    # Input Frame In Model
    def setFrame(self, frame):
        results = self.model.predict(source=[frame], save=False, verbose=False)[0]
        self.results = []
        self.inputDS = []
        label = None
        xmin, ymin, xmax, ymax = None, None, None, None
        # Read Boxes Data
        for data in results.boxes.data.tolist():
            confidence = float(data[4])
            if confidence < self.confidenceThreshold:
                continue
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            label = int(data[5])
            self.results.append([xmin, ymin, xmax, ymax, label])
            self.inputDS.append([[xmin, ymin, xmax-xmin, ymax-ymin], confidence, label])
        # DeepSort
        if self.DS == True:
            self.getDeepSort(frame)

    # Get Vehicle Box Data
    def getData(self):
        return self.results
    
    # Draw And Get Frame
    def getDraw(self, orgFrame):
        frame = np.copy(orgFrame)
        # Draw Boxes In Frame
        for xmin, ymin, xmax, ymax, label in self.results:
            if None in [xmin, ymin, xmax, ymax, label]:
                continue
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), self.green, 2)
            textboxSize = len(self.classList[label])*11
            cv2.rectangle(frame, (xmin, ymin-20), (xmin+textboxSize, ymin), self.green, -1)
            cv2.putText(frame, self.classList[label], (xmin+5, ymin-8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.white, 2)
        # return 
        return frame
    
    # Set Deep Sort
    def setDeepSort(self, DSage):
        self.DSage = DSage
        self.tracker = DeepSort(self.DSage)

    # Deep Sort In Boxes
    def getDeepSort(self, frame):
        newResults = []
        tracks = self.tracker.update_tracks(self.inputDS, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue
            ltrb = track.to_ltrb()
            newResults.append([int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3]), track.get_det_class()])
        self.results = newResults    