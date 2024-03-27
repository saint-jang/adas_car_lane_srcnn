# last day : 2024-02-13-11-16

import cv2
import numpy as np
import argparse
import datetime
import curses
import matplotlib.pyplot as plt
from ADASlib.VehicleDetection import VD_Model
from ADASlib.LaneDetection import LD_Model

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
parser.add_argument("--recordName", dest="recordName", type=str, default="adasVideo.mp4")
parser.add_argument("--graphFPS", dest="graphFPS", type=bool, default=False)
parser.add_argument("--graphFPSname", dest="graphFPSname", type=str, default="adasGraphFPS.png")
parser.add_argument("--noCurses", dest="noCurses", type=bool, default=False)
parser.add_argument("--camNumber", dest="camNumber", type=int, default=0)
args = parser.parse_args()

# Vehicle Detection Model Option
NO_VD = args.noVD
VD_MODEL = args.VDmodel
VD_MODEL_ADD = args.VDmodelAdd
DEEP_SORT = args.VDdeepsort
VDmodel = VD_Model(model=VD_MODEL, modelAdd=VD_MODEL_ADD, DS=DEEP_SORT)

# Lane Detection Model Option
NO_LD = args.noLD
LD_MODEL = args.LDmodel
LD_MODEL_ADD = args.LDmodelAdd
LDmodel = LD_Model(LD_MODEL, LD_MODEL_ADD)

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

# Write VIdeo Name
RECORD_NAME = args.recordName

# GraphFPS On Off
GRAPH_FPS = args.graphFPS

# GraphFPS Name
GRAPH_FPS_NAME = args.graphFPSname

# Curses On Off
NO_CURSES = args.noCurses

# Camera Number
CAM_NUMBER = args.camNumber
        
# Frame View Function
def viewFrame(cap, stdscr):
    out = None
    # Write Frame Option
    if (RECORD == True) and (INPUT_TYPE != "image"):
        h, w = int(cap.get(4)), int(cap.get(3))
        if OUTPUT_SIZE != 1.0:
            h = int(h * OUTPUT_SIZE)
            w = int(w * OUTPUT_SIZE)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(RECORD_NAME, fourcc, 30.0, (w, h))

    # GraphFPS set
    graphFPSxCount = 0
    graphFPSx = []
    graphFPSy = []

    # Play Frames
    while True:
        # Write Curses Texts
        texts = []
        texts.append(f"Loop : {LOOP}")
        texts.append(f"Skip : {SKIP}")
        texts.append(f"Input : {INPUT}")
        texts.append(f"Record : {RECORD}")
        texts.append(f"graphFPS : {GRAPH_FPS}")

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
            if INPUT_TYPE == "video":
                texts.append(f"VideoFrame : {int(cap.get(cv2.CAP_PROP_POS_FRAMES))} / {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
        else:
            # Image Read
            frame = cv2.imread(cap, cv2.IMREAD_COLOR)

        texts.append(f"InputSize : {frame.shape[1]} {frame.shape[0]}")

        # graphFPSxCount set
        if GRAPH_FPS == True:
            graphFPSxCount += 1
            if SKIP != 0:
                graphFPSxCount += SKIP
            graphFPSx.append(graphFPSxCount)

        # Resize Input Frame Size
        frame = cv2.resize(frame, (0, 0), fx=INPUT_SIZE, fy=INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        texts.append(f"InputResize : {frame.shape[1]} {frame.shape[0]}")

        # On Off Use Model
        if NO_MODEL == False:
            copyFrame = np.copy(frame)
            if NO_LD == False:
                # Lane Dtaction
                LDmodel.setFrame(copyFrame)
                # Draw Lane
                frame = LDmodel.getDraw(frame)
            if NO_VD == False:
                # Vehicle Detection
                VDmodel.setFrame(copyFrame)
                # Draw Vehicle Baxes
                frame = VDmodel.getDraw(frame)

        # Resize Output Frame Size
        frame = cv2.resize(frame, (0, 0), fx=OUTPUT_SIZE, fy=OUTPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        texts.append(f"OutputResize : {frame.shape[1]} {frame.shape[0]}")

        # End Time Read
        end_time = datetime.datetime.now()

        # Work Time Read
        work_time = (end_time - start_time).total_seconds()

        # Draw FPS in Frame
        fps = 1 / work_time
        cv2.putText(frame, f"FPS : {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        texts.append(f"FPS : {fps:.2f}")
        if GRAPH_FPS == True:
            graphFPSy.append(fps)

        # Write Frame 
        if (RECORD == True) and (INPUT_TYPE != "image"):
            out.write(frame)

        # Show NO_CURSES
        if NO_CURSES ==False:
            # Show Curses
            cursesDisplay(stdscr, texts)
        
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
    
    # graphFPSwrite
    if GRAPH_FPS == True:
        plt.plot(graphFPSx, graphFPSy)
        plt.title(GRAPH_FPS_NAME)
        plt.xlabel("Frame")
        plt.ylabel("FPS")
        plt.savefig(GRAPH_FPS_NAME)

    # Clearing Code
    if (RECORD == True) and (INPUT_TYPE != "image"):
        out.release()
    if INPUT_TYPE != "image":
        cap.release()
    cv2.destroyAllWindows()

# Image View Function
def viewImage(stdscr):
    viewFrame(INPUT, stdscr)

# Video View Function
def viewVideo(stdscr):
    cap = cv2.VideoCapture(INPUT)
    if cap.isOpened()==False:
        print("Video Open Failed")
        exit()
    viewFrame(cap, stdscr)

# Camera View Function
def viewCamera(stdscr):
    cap = cv2.VideoCapture(CAM_NUMBER)
    if cap.isOpened()==False:
        print("Camera Open Failed")
        exit()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    viewFrame(cap, stdscr)

# Curses Display
def cursesDisplay(stdscr, texts=[]):
    stdscr.clear()
    for i, text in enumerate(texts):
        stdscr.addstr(i, 0, text)
    stdscr.refresh()

# main
def main(stdscr):
    if INPUT_TYPE == "image":
        viewImage(stdscr)
    elif INPUT_TYPE == "video":
        viewVideo(stdscr)
    elif INPUT_TYPE == "camera":
        viewCamera(stdscr)

if __name__ == "__main__":
    if NO_CURSES == False:
        curses.wrapper(main)
    else:
        main(None)