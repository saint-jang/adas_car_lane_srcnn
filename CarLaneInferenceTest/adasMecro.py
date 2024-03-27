import os
import glob
import argparse

# Program Option
parser = argparse.ArgumentParser()
parser.add_argument("--inputVideo", dest="inputVideo", type=str, default=None)
parser.add_argument("--inputVD", dest="inputVD", type=str, default=None)
parser.add_argument("--inputLD", dest="inputLD", type=str, default=None)
parser.add_argument("--address", dest="address", type=str, default="")
parser.add_argument("--mecro", dest="mecro", type=bool, default=False)
args = parser.parse_args()

# First Option
mainStr = "python3 adasViewer.py"
mainStr += " --noCurses True"
mainStr += " --record True"
mainStr += " --inputType video"
mainStr += " --loop False"

# Input Video Option
INPUT_VIDEO = args.inputVideo
inputVideo = None
if INPUT_VIDEO == None:
    print("--inputVideo None")
    exit()
else:
    inputVideo = glob.glob(INPUT_VIDEO)
    if inputVideo == []:
        print("--inputVideo []")
        exit()

# Input VDmodel Option
INPUT_VD = args.inputVD
NO_VD = False
if INPUT_VD == None:
    NO_VD = True
else:
    inputVD = glob.glob(INPUT_VD)
    if inputVD == []:
        NO_VD = True

# Input LDmodel Option
INPUT_LD = args.inputLD
NO_LD = False
if INPUT_LD == None:
    NO_LD = True
else:
    inputLD = glob.glob(INPUT_LD)
    if inputLD == []:
        NO_LD = True

# Output Address
ADDRESS = args.address

# mecro On Off
MECRO = args.mecro

# if noModel, exit()
if (NO_VD == True) and (NO_LD == True):
    print("NO model")
    exit()

# Mecro Start
if NO_VD == False:
    videoType = ".mp4"
    for vdNum, VD in enumerate(inputVD):
        for videoNum, video in enumerate(inputVideo):
            VDstr = ""
            VDstr += mainStr
            VDstr += " --noLD True"
            VDstr += " --VDmodel "
            VDstr += VD
            VDstr += " --input "
            VDstr += video
            VDstr += " --recordName "
            outputVD = VD.split(".")[0]
            outputVD = outputVD.split("/")
            outputVD = outputVD[len(outputVD) - 1]
            video = video.split(".")[0]
            video = video.split("/")
            video = video[len(video) - 1]
            outputName = ADDRESS + "VD_" + outputVD + "_" + video + videoType
            VDstr += outputName
            if MECRO == True:
                os.system(VDstr)
            else:
                print(VDstr)

if NO_LD == False:
    videoType = ".mp4"
    for ldNum, LD in enumerate(inputLD):
        for videoNum, video in enumerate(inputVideo):
            LDstr = ""
            LDstr += mainStr
            LDstr += " --noVD True"
            LDstr += " --LDmodel "
            LDstr += LD
            LDstr += " --input "
            LDstr += video
            LDstr += " --recordName "
            outputLD = LD.split(".")[0]
            outputLD = outputLD.split("/")
            outputLD = outputLD[len(outputLD) - 1]
            video = video.split(".")[0]
            video = video.split("/")
            video = video[len(video) - 1]
            outputName = ADDRESS + "LD_" + outputLD + "_" + video + videoType
            LDstr += outputName
            if MECRO == True: 
                os.system(LDstr)
            else:
                print(LDstr)