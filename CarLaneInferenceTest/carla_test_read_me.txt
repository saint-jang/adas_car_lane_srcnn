CARLA 시뮬레이션에서 작동하는 파일 추가

파일 이름 : manual_control.py

설치방법
1. 콘다 설치

2. 파이썬 3.8.0 가상환경 생성 및 접속
conda create --name carla python=3.8.0
conda activate carla
manual_control.py파일이 있는 위치로 이동

3. 종속성 라이브러리 확인 및 설치
manual_control.py 가 있는 위치에 다음 파일들이 있어야 함
ADASlib/LaneChange.py
ADASlib/LaneDetection.py
ADASlib/VehicleDetection.py
epoch_200_batch_50_loss_best.onnx
mycoco128.txt
myUltrafastLaneDetector.py
yolov8n_epoch_200_batch_60_best.pt
다음 명령어로 종속성 라이브러리 설치 (파이썬 3.8.0일것)(넘파이버전이 1.18.4이상이면 지울것)
pip install ultralytics onnx onnxruntime deep-sort-realtime
pip install -r requirements.txt

4. 서버 실행상태 및 IP 확인
303강의실 hkit303 와이파이에 접속.
manual_control.py 코드에서 서버 IP를 192.168.0.2로 변경 (이미 변경되어 있음)
포트 2000번 확인

5. 프로그램 실행
python3 manual_control.py
