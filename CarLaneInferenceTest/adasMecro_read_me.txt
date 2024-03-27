adasMecro.py 사용법

설명
	adasViewer.py에서 많은 테스트 결과를 쉽게 얻기 위해 만든 프로그램.
	차량인식, 차선인식을 동시에 사용한 영상은 생성할 수 없음 (차량인식 또는 차선인식)

옵션 
	--inputVideo <테스트동영상폴더/*>
		테스트로 사용할 동영상 파일이 모인 폴더의 모든 동영상을 사용

	--inputVD <차량인식모델폴더/*>
		차량인식 모델이 모인 폴더의 모든 모델 사용
		해당 옵션을 사용하지 않으면 차량인식 결과 영상을 생성하지 않음

	--inputLD <차선인식모델폴더/*>
		차선인식 모델이 모인 폴더의 모든 모델 사용
		해당 옵션을 사용하지 않으면 차선인식 결과 영상을 생성하지 않음

	--address <저장위치/>
		결과 파일의 저장 위치
		기본설정 : 프로그램 실행 위치

	--mecro <False/True>
		True : 결과 파일 생성
		False : 명령어를 출력 (명령어가 올바르게 입력되는지 디버깅하기 위한 용도)
		기본설정 : False

사용 예시
python3 adasMecro.py --inputVideo "Video/*" --inputVD "VDmodel/*" --inputLD "LDmodel/*" --address "ResultVideo/" --mecro True
