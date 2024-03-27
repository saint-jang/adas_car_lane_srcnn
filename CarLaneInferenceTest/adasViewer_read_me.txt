adasViewer.py 사용법

필요한 라이브러리(버전은 제작자 환경 기준으로 작성됨)
	python(3.8.10)
	ultralytics(8.1.1)
	onnx(1.15.0)
	onnxruntime(1.16.3)
	deep-sort-realtime(1.3.2)
	
라이브러리 설치 방법
	pip install ultralytics
	pip install onnx
	pip install onnxruntime
	pip install deep-sort-realtime

터미널에서 사용하는 방법
	기본 설정 사용법(함께 준비된 예제 동영상과 모델들을 사용)
		python adasViewer.py
		python3 adasViewer.py
	
옵션 사용법
	python3 adasViewer.py <--옵션> <옵션값>

옵션 종류
	차량인식 관련
		--noVD <Fasle/True>
			차량인식 모델 사용 여부
			기본설정 : False
		--VDmodel <차량인식모델>
			차량인식 모델 변경
			기본설정 : "yolov8n_epoch_200_batch_60_best.pt"
		--VDmodelAdd <추가사항>
			차량인식 모델의 추가사항 변경
			기본설정 : "mycoco128.txt"
		--VDdeepsort <False/True>
			차량인식 모델에서 추적알고리즘 적용 여부
			기본설정 : True
	차선인식 관련
		--noLD <False/True>
			차선인식 모델 사용 여부
			기본설정 : False
		--LDmodel <차선인식모델>
			차선인식 모델 변경
			기본설정 : "epoch_200_batch_50_loss_best.onnx"
		--LDmodelAdd <추가사항>
			차선인식 모델의 추가사항 변경
			기본설정 : "culane"
	차선유지 관련
		--LaneChange <Flase/True>
			차선유지 경고메시지 사용 여부 (차선인식 모델이 사용중일때 사용가능)
			기본설정 : True
	입력 데이터 관련
		--input <입력데이터>
			입력 데이터 변경
			기본설정 : "testVideo.mp4"
		--inputType <데이터타입>
			입력 데이터 타입 설정 ["image", "video", "camera"]
			기본설정 : "video"
		--inputSize <적용비율>
			모델에 적용할 데이터 크기 변경
			기본설정 : 1.0
		--outputSize <적용비율>
			출력할 데이터 크기 변경
			기본설정 : 1.0
		--skip <스킵 수>
			비디오, 카메라에서 스킵할 프레임 수 설정
			기본설정 : 0
		--camNumber <카메라번호>
			카메라 번호 설정
			기본설정 : 0
	동영상 저장 옵션
		--record <False/True>
			동영상 저장 여부, 이미지, 비디오, 카메라에서 모두 동영상으로 저장
			기본설정 : False
		--recordName <파일이름>
			저장될 동영상 파일이름 변경
			기본설정 : "adasVideo.mp4"
	FPS 그래프 저장 옵션
		--graphFPS <False/True>
			FPS그래프 출력 여부
			기본설정 : False
		--graphFPSname <파일이름>
			저장될 그래프 파일이름 변경
			기본설정 : "adasGraphFPS.png"
	기타 옵션
		--loop <False/True>
			비디오에서 영상 무한재생 여부
			기본설정 : False
		--noModel <False/True>
			모델 적용 여부, False면 입력데이터를 그대로 출력
			비디오, 카메라의 경우 재생속도 25FPS고정
			기본설정 : False
		--noCurses <False/True>
			curses라이브러리를 이용해 내부 파라미터 값 출력
			기본설정 : True
		--noView <False/True>
			영상화면 사용 여부
			기본설정 : True