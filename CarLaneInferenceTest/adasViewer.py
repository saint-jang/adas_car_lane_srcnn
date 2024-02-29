# last day : 2024-02-08 14:12

import cv2
import numpy as np
import argparse
import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from myUltrafastLaneDetector import UltrafastLaneDetector, ModelType 

## input Parameter ##
# Option Read
parser = argparse.ArgumentParser()
parser.add_argument("--noVD", dest="noVD", type=bool, default=False)
parser.add_argument("--VDmodel", dest="VDmodel", type=str, default='best.pt')
parser.add_argument("--VDmodelAdd", dest="VDmodelAdd", type=str, default='mycoco128.txt')
parser.add_argument("--VDdeepsort", dest="VDdeepsort", type=bool, default=True)
parser.add_argument("--noLD", dest="noLD", type=bool, default=False)
parser.add_argument("--LDmodel", dest="LDmodel", type=str, default="epoch_20_batch_32.onnx")
parser.add_argument("--LDmodelAdd", dest="LDmodelAdd", type=str, default="culane")
parser.add_argument("--input", dest="input", type=str, default="testVideo.mp4")
parser.add_argument("--inputType", dest="inputType", type=str, default="video")
parser.add_argument("--skip", dest="skip", type=int, default=0)
parser.add_argument("--loop", dest="loop", type=str, default="False")
parser.add_argument("--noModel", dest="noModel", type=bool, default=False)
parser.add_argument("--inputSize", dest="inputSize", type=float, default=1.0)
parser.add_argument("--outputSize", dest="outputSize", type=float, default=1.0)
parser.add_argument("--record", dest="record", type=bool, default=False)
args = parser.parse_args()

# Vehicle Detection Model Option
NO_VD = args.noVD
VD_MODEL = args.VDmodel
VD_MODEL_ADD = args.VDmodelAdd
coco128 = open(VD_MODEL_ADD, 'r')
VDdata = coco128.read()
class_list = VDdata.split('\n')
coco128.close()
VDmodel = YOLO(VD_MODEL)

# Deep Sort On Off Option
DEEP_SORT = args.VDdeepsort
if DEEP_SORT == True:
    VDtracker = DeepSort(max_age=50)

# Lane Detection Model Option
NO_LD = args.noLD
LD_MODEL = args.LDmodel
LD_MODEL_ADD = args.LDmodelAdd
if LD_MODEL_ADD == "culane":
    LD_MODEL_ADD = ModelType.CULANE
elif LD_MODEL_ADD == "tusimple":
    LD_MODEL_ADD = ModelType.TUSIMPLE
else:
    print(f"can't input modelType : {LD_MODEL_ADD}")
    exit()
LDmodel = UltrafastLaneDetector(model_path=LD_MODEL, model_type=LD_MODEL_ADD)

# Input Data Address
INPUT = args.input 

# Input Data Type [ image, video, camera ]
INPUT_TYPE = args.inputType 
if INPUT_TYPE not in ["video", "image", "camera"]:
    print(f"can't input inputType : {INPUT_TYPE}")
    exit()

# Frame Skip Count
SKIP = args.skip 

# Video Loop On Off
LOOP = False
if args.loop == "True":
    LOOP = True

# Model On Off
NO_MODEL = args.noModel

# Resize Input Data Size
INPUT_SIZE = args.inputSize

# Resize Output Data Size
OUTPUT_SIZE = args.outputSize

# Record On Off
RECORD = args.record

# Vehicle Detection Function
def VDmodelFunction(frame):
    detection = VDmodel.predict(source=[frame], save=False, verbose=False)[0]
    return detection

# Draw Vehicle Boxes
def drawBox(frame, detection):
    CONFIDENCE_THRESHOLD = 0.6
    GREEN = (0, 255, 0)
    WHITE = (255, 255, 255)

    results = []
    label = None
    xmin, ymin, xmax, ymax = None, None, None, None
    # Read Boxes Data
    for data in detection.boxes.data.tolist():
        confidence = float(data[4])
        if confidence < CONFIDENCE_THRESHOLD:
            continue
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        label = int(data[5])
        results.append([[xmin, ymin, xmax-xmin, ymax-ymin], confidence, label])

    # Deep Sort Code => Change Boxes Chase Data
    if DEEP_SORT == True:
        tracks = VDtracker.update_tracks(results, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])

    # Draw Boxes in Frame
    for data in detection.boxes.data.tolist(): 
        if None in [xmin, ymin, xmax, ymax, label]:
            continue
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
        textboxSize = len(class_list[label]) * 11
        cv2.rectangle(frame, (xmin, ymin - 20), (xmin + textboxSize, ymin), GREEN, -1)
        cv2.putText(frame, class_list[label], (xmin + 5, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)
    
    return frame
        
# Lane Detection Function
def LDmodelFunction(frame):
    orgImgH, orgImgW = frame.shape[:2]
    cfgImgH, cfgImgW = float(LDmodel.cfg.img_h), float(LDmodel.cfg.img_w)
    _, results, laneDetect = LDmodel.detect_lanes(frame)
    for n1, lane in enumerate(results):
        for n2, point in enumerate(lane):
            changeW = int((orgImgW * point[0]) / cfgImgW)
            changeH = int((orgImgH * point[1]) / cfgImgH)
            results[n1][n2] = [changeW, changeH]

    return results, laneDetect

# Draw Lane in Frame
def drawLane(frame, results, laneDetect):
    lane_colors = [(0,0,255),(0,255,0),(255,0,0),(0,255,255)]

    # Draw My Driving Road
    if(laneDetect[1] and laneDetect[2]):
        copyFrame = np.copy(frame)
        cv2.fillPoly(copyFrame, pts = [np.vstack((results[1],np.flipud(results[2])))], color =(255,191,0))
        frame = cv2.addWeighted(frame, 0.7, copyFrame, 0.3, 0)

    # Draw Lane Points
    for lane_num, lane_points in enumerate(results):
                for lane_point in lane_points:
                    cv2.circle(frame, (lane_point[0],lane_point[1]), 3, (lane_colors[lane_num]), -1)
    
    return frame

# Frame View Function
def viewFrame(cap):
    out = None
    # Write Frame Option
    if (RECORD == True) and (INPUT_TYPE != "image"):
        h, w = int(cap.get(4)), int(cap.get(3))
        if OUTPUT_SIZE != 1.0:
            h = int(h * OUTPUT_SIZE)
            w = int(w * OUTPUT_SIZE)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('adasVideo.mp4', fourcc, 30.0, (w, h))
        print(w, h)

    # Play Frames
    while True:
        # Start Time Read
        start_time = datetime.datetime.now()

        # Input Type Is Image, Use Funcion cv2.imread(cap) 
        # Input Type Is Video Or Camera, Use Function cap.read()
        frame = None
        if INPUT_TYPE != "image":
            for i in range(0, SKIP + 1): # Frame Skip
                ret, frame = cap.read() # Frame Read

                # Loop On Off
                if LOOP == False:
                    if ret == False:
                        print("Frame End")
                        cap.release()
                        cv2.destroyAllWindows()
                        exit()
                else:
                    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        else:
            # Image Read
            frame = cv2.imread(cap, cv2.IMREAD_COLOR)

        # Resize Input Frame Size
        frame = cv2.resize(frame, (0, 0), fx=INPUT_SIZE, fy=INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        
        # On Off Use Model
        if NO_MODEL == False:
            copyFrame = np.copy(frame)
            if NO_LD == False:
                # Lane Dtaction
                LDdatas, LD = LDmodelFunction(copyFrame)
                # Draw Lane
                frame = drawLane(frame, LDdatas, LD)
            if NO_VD == False:
                # Vehicle Detection
                VDdatas = VDmodelFunction(copyFrame)
                # Draw Vehicle Baxes
                frame = drawBox(frame, VDdatas)

        # Resize Output Frame Size
        frame = cv2.resize(frame, (0, 0), fx=OUTPUT_SIZE, fy=OUTPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        
        # End Time Read
        end_time = datetime.datetime.now()

        # Work Time Read
        work_time = (end_time - start_time).total_seconds()

        # Draw FPS in Frame
        fps = f"FPS : {1 / work_time:.2f}"
        cv2.putText(frame, fps, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Write Frame 
        if (RECORD == True) and (INPUT_TYPE != "image"):
            out.write(frame)

        # Show Frame
        cv2.imshow('adasViewer.py', frame)

        # If No Use Model, Frame Speed Is 25
        timeRate = 25
        if NO_MODEL == False:
            timeRate = 1
        
        # If Input 'Q'Key, Stop Play Frames
        if cv2.waitKey(timeRate) & 0xFF == ord('q'):
            print("Frame Close")
            break

    # Clearing Code
    if (RECORD == True) and (INPUT_TYPE != "image"):
        out.release()
    if INPUT_TYPE != "image":
        cap.release()
    cv2.destroyAllWindows()

# Image View Function
def viewImage():
    viewFrame(INPUT)

# Video View Function
def viewVideo():
    cap = cv2.VideoCapture(INPUT)
    if cap.isOpened()==False:
        print("Video Open Failed")
        exit()
    viewFrame(cap)

# Camera View Function
def viewCamera():
    cap = cv2.VideoCapture(0)
    if cap.isOpened()==False:
        print("Camera Open Failed")
        exit()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    viewFrame(cap)

# main
if __name__ == "__main__":
    if INPUT_TYPE == "image":
        viewImage()
    elif INPUT_TYPE == "video":
        viewVideo()
    elif INPUT_TYPE == "camera":
        viewCamera()