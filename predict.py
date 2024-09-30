"""
******************************************************************************************
 * FileName      : prediect.py
 * Description   : Function to Perform Model Prediction and Organize Results
 * Author        : Jeong Yoo Lim
 * Last modified : 2024.09.30
 ******************************************************************************************
"""
import torch
import cv2
from ultralytics import YOLO

# YOLOv8 모델 불러오기
model = YOLO('best.pt')

# 웹캠 객체 생성
cap = cv2.VideoCapture(2)

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    
    if not ret:
        print("카메라에서 영상을 가져올 수 없습니다.")
        break

    # YOLOv8 모델 적용 (프레임에서 객체 감지)
    results = model(frame)

    # 감지된 객체를 프레임 위에 표시
    frame = results[0].plot()  # YOLOv8에서 plot() 메서드를 사용하여 결과 렌더링

    # 웹캠 프레임 출력
    cv2.imshow('YOLOv8 Webcam', frame)
    
    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠과 모든 창 종료
cap.release()
cv2.destroyAllWindows()
