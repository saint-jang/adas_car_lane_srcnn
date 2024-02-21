import cv2
import numpy as np
import argparse
import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from myUltrafastLaneDetector import UltrafastLaneDetector, ModelType 

'''옵션 설정'''
# 옵션 읽어들이기
parser = argparse.ArgumentParser()
parser.add_argument("--noVD", dest="noVD", type=bool, default=False)
parser.add_argument("--VDmodel", dest="VDmodel", type=str, default='best.pt')
parser.add_argument("--VDmodelAdd", dest="VDmodelAdd", type=str, default='mycoco128.txt')
parser.add_argument("--VDdeepsort", dest="VDdeepsort", type=bool, default=False)
parser.add_argument("--noLD", dest="noLD", type=bool, default=False)
parser.add_argument("--LDmodel", dest="LDmodel", type=str, default="epoch_20_batch_32.onnx")
parser.add_argument("--LDmodelAdd", dest="LDmodelAdd", type=str, default="culane")
parser.add_argument("--input", dest="input", type=str, default="testVideo.mp4")
parser.add_argument("--inputType", dest="inputType", type=str, default="video")
parser.add_argument("--skip", dest="skip", type=int, default=0)
parser.add_argument("--loop", dest="loop", type=bool, default=True)
parser.add_argument("--noModel", dest="noModel", type=bool, default=False)
parser.add_argument("--inputSize", dest="inputSize", type=float, default=1.0)
parser.add_argument("--outputSize", dest="outputSize", type=float, default=1.0)
args = parser.parse_args()

# 차량인식 모델 설정
NO_VD = args.noVD
VD_MODEL = args.VDmodel
VD_MODEL_ADD = args.VDmodelAdd
coco128 = open(VD_MODEL_ADD, 'r')
VDdata = coco128.read()
class_list = VDdata.split('\n')
coco128.close()
VDmodel = YOLO(VD_MODEL)

# 차량인식 모델에서 deppsort 기능 설정
DEEP_SORT = args.VDdeepsort
if DEEP_SORT == True:
    VDtracker = DeepSort(max_age=50)

# 차선인식 모델 설정
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

# 모델을 적용할 입력 데이터 이름 또는 주소
INPUT = args.input 

# 입력 데이터 타입 (이미지, 비디오, 카메라)
INPUT_TYPE = args.inputType 
if INPUT_TYPE not in ["video", "image", "camera"]:
    print(f"can't input inputType : {INPUT_TYPE}")
    exit()

# 비디오, 카메라에 모델 적용시 스킵되는 프레임 이미지 수
SKIP = args.skip 

# 비디오에 모델 적용시 반복재생 여부
LOOP = args.loop 

# 모델을 적용하지 않고 출력
NO_MODEL = args.noModel

# 입력되는 영상 크기 설정
INPUT_SIZE = args.inputSize

# 출력되는 영상 크기 설정
OUTPUT_SIZE = args.outputSize

'''모델 관련 함수 모음'''
# 차량을 인식하여 박스로 표현
def VDmodelFunction(frame):
    # 모델에 이미지 적용
    detection = VDmodel.predict(source=[frame], save=False, verbose=False)[0]
    return detection

# 차량 박스 그리는 함수
def drawBox(frame, detection):
    # 해당 값보다 적은 정확도를 가진 객체는 표시하지 않음
    CONFIDENCE_THRESHOLD = 0.6

    # 색 설정
    GREEN = (0, 255, 0)
    WHITE = (255, 255, 255)

    # 검출된 객체 데이터를 박스데이터로 변환 후 저장
    results = []
    label = None
    for data in detection.boxes.data.tolist():

        # 정확도가 설정값보다 적으면 기록하지 않음
        confidence = float(data[4])
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        # 박스 그릴 포인트값 저장
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        # 객체 종류 저장
        label = int(data[5])
        results.append([[xmin, ymin, xmax-xmin, ymax-ymin], confidence, label])

    # DEEP_SORT가 True이면 객체추적 알고리즘을 적용
    if DEEP_SORT == True:
        tracks = VDtracker.update_tracks(results, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()

            # 객체추적 알고리즘이 적용된 박스데이터로 변경
            xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])

    # 이미지에 정보 그리기
    for data in detection.boxes.data.tolist(): 
        # 박스 그리기
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
        textboxSize = len(class_list[label]) * 11
        cv2.rectangle(frame, (xmin, ymin - 20), (xmin + textboxSize, ymin), GREEN, -1)

        # 객체 정보 그리기
        cv2.putText(frame, class_list[label], (xmin + 5, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)
    
    return frame
        
# 차선을 인식하여 점으로 표현
def LDmodelFunction(frame):
    # 원본 이미지의 크기, 차선인식 모델의 가중치에 의해 변화된 이미지의 크기 가져오기
    orgImgH, orgImgW = frame.shape[:2]
    cfgImgH, cfgImgW = float(LDmodel.cfg.img_h), float(LDmodel.cfg.img_w)

    # 차선인식 모델에서 데이터 추출
    _, results, laneDetect = LDmodel.detect_lanes(frame)

    # 모델에서 뽑은 차선 데이터가 우리가 원하는 데이터보다 일그러져 있으므로 후처리
    for n1, lane in enumerate(results):
        for n2, point in enumerate(lane):
            changeW = int((orgImgW * point[0]) / cfgImgW)
            changeH = int((orgImgH * point[1]) / cfgImgH)
            results[n1][n2] = [changeW, changeH]

    # 차선 데이터 번환
    return results, laneDetect

# 차선 그리는 함수
def drawLane(frame, results, laneDetect):
    # 차선 색 선언
    lane_colors = [(0,0,255),(0,255,0),(255,0,0),(0,255,255)]

    # 사다리꼴로 주행중인 차선 표현
    if(laneDetect[1] and laneDetect[2]):
        copyFrame = np.copy(frame)
        cv2.fillPoly(copyFrame, pts = [np.vstack((results[1],np.flipud(results[2])))], color =(255,191,0))
        frame = cv2.addWeighted(frame, 0.7, copyFrame, 0.3, 0)

    # 점으로 차선 표현
    for lane_num, lane_points in enumerate(results):
                for lane_point in lane_points:
                    cv2.circle(frame, (lane_point[0],lane_point[1]), 3, (lane_colors[lane_num]), -1)
    
    # 그린 이미지 반환
    return frame

'''영상 출력 함수 모음'''
# 프레임 출력 함수, 비디오, 카메라에서 사용
def viewFrame(cap):
    # 영상 화면 재생
    while True:
        # 시작시간 기록
        start_time = datetime.datetime.now()

        # 이미지면 cv2.imread()를, 비디오와 카메라면 cap.read()를 사용하여 이미지 읽기
        frame = None
        if INPUT_TYPE != "image":
            for i in range(0, SKIP + 1): # SKIP 수 만큼 스킵
                ret, frame = cap.read()

                # 영상 이미지 읽기를 실패하면 화면 재생 종료
                if ret == False:
                    print("Frame Error")
                    break

                # LOOP가 False면 영상이 끝나면 종료
                if LOOP == False:
                    if ret == False:
                        print("Frame End")
                        cap.release()
                        cv2.destroyAllWindows()
                        exit()
                # LOOP가 True면 영상이 끝나도 반복 재생
                else:
                    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            # 이미지 읽기
            frame = cv2.imread(cap, cv2.IMREAD_COLOR)

        # 입력 이미지 크기 변환, OUTPUT_SIZE가 1.0 이면 적용하지 않음
        if INPUT_SIZE != 1.0:
            frame = cv2.resize(frame, (0, 0), fx=INPUT_SIZE, fy=INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        
        # NO_MODEL이 False면 모델을 적용, True면 모델을 적용하지 않음
        if NO_MODEL == False:
            # 이미지 복사
            copyFrame = np.copy(frame)

            if NO_LD == False:
                # 차선인식 데이터 추출
                LDdatas, LD = LDmodelFunction(copyFrame)

                # 이미지에 차선 표시
                frame = drawLane(frame, LDdatas, LD)

            if NO_VD == False:
                # 차량인식 데이터 추출
                VDdatas = VDmodelFunction(copyFrame)

                # 이미지에 차량 표시
                frame = drawBox(frame, VDdatas)

        # 출력 이미지 크기 변환, OUTPUT_SIZE가 1.0 이면 적용하지 않음
        if OUTPUT_SIZE != 1.0:
            frame = cv2.resize(frame, (0, 0), fx=OUTPUT_SIZE, fy=OUTPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        
        # 종료시간 기록
        end_time = datetime.datetime.now()

        # 사용된 시간 기록
        work_time = (end_time - start_time).total_seconds()

        # 이미지에 FPS 표시
        fps = f"FPS : {1 / work_time:.2f}"
        cv2.putText(frame, fps, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


        # 이미지를 화면에 출력
        cv2.imshow('adasViewer.py', frame)

        # 모델적용이 없을 시 영상 속도를 25로 변경
        timeRate = 25
        if NO_MODEL == False:
            timeRate = 1
        
        # 'Q'가 입력되면 비디오 출력을 종료
        if cv2.waitKey(timeRate) & 0xFF == ord('q'):
            print("Frame Close")
            break

    # 코드 마무리
    if INPUT_TYPE != "image":
        cap.release()
    cv2.destroyAllWindows()

# 이미지 출력 함수
def viewImage():
    # 영상 화면 재생
    viewFrame(INPUT)

# 비디오 출력 함수
def viewVideo():
    # 비디오 열기
    cap = cv2.VideoCapture(INPUT)

    # 비디오 열기에 실패하면 종료
    if cap.isOpened()==False:
        print("Video Open Failed")
        exit()

    # 영상 화면 재생
    viewFrame(cap)

# 카메라 출력 함수
def viewCamera():
    # 카메라 열기
    cap = cv2.VideoCapture(0)

    # 카메라 열기에 실패하면 프로그램 종료
    if cap.isOpened()==False:
        print("Camera Open Failed")
        exit()
    
    # 카메라 셋팅 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 영상 화면 재생
    viewFrame(cap)

'''메인'''
if __name__ == "__main__":
    if INPUT_TYPE == "image":
        viewImage()
    elif INPUT_TYPE == "video":
        viewVideo()
    elif INPUT_TYPE == "camera":
        viewCamera()