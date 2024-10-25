from flask import Flask, jsonify, render_template, Response, request
import cv2
import threading
import os
import time
from ultralytics import YOLO
from flask_cors import CORS
import requests
from flasgger import Swagger
from datetime import datetime
import torch
from torchvision.ops import nms


app = Flask(__name__)
CORS(app)
swagger = Swagger(app)

model1 = YOLO('abnormal.pt')
model2 = YOLO('growth.pt')

# 웹캠 인덱스
camera = cv2.VideoCapture(2)

# 이미지 저장 경로
img_path1 = './static/captured_image_model1.jpg'
img_path2 = './static/captured_image_model2.jpg'

# 파일 접근을 보호하기 위한 Lock 생성
lock = threading.Lock()

# 클래스 이름과 색상 정의 (각 모델에 대해)
class_names_model1 = {
    0: 'hole',
    1: 'wither'
}
class_names_model2 = {
    0: 'level_1',
    1: 'level_2',
    2: 'level_3',
}

class_colors_model1 = {
    0: (255, 0, 0),  # 빨간색 (구멍)
    1: (127, 255, 212),  # 민트색 (시든거)
}

class_colors_model2 = {
    0: (0, 0, 255),
    1: (0, 255, 0),
    2: (255, 0, 255),
}


status = False # 감지 상태 저장
freeze_status = False # 상태 정보 고정
status_lock = threading.Lock() # 동시 접근 제어를 위한 lock

box_count_history = []  # 바운딩 박스 개수를 저장하는 리스트

def send_notification(currnet_box_count):
    url = "http://3.34.153.235:8080/api/notification/save"
    params = {
        'userId': 1,
        'box_count': currnet_box_count,
        'timestamp': datetime.now().isoformat()
    }

    print(f"Sending GET request to {url} with params: {params}")

    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            print("Notification sent successfully.")
        else:
            print(f"Failed to send notification: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error sending notification: {e}")

def abnormal(model, img_path, class_names, class_colors, sleep_time):
    global status, freeze_status, box_count_history, current_box, class_counts
    current_box = []

    while True:
        with lock:
            camera.grab()
            ret, frame = camera.retrieve()

        if ret:
            results = model(frame)
            detected_counts = {name: 0 for name in class_names.values()}
            current_box_count = 0
            class_counts = {0: 0, 1: 0}
            

            # 감지된 객체 처리
            for result in results:
                boxes = result.boxes
                current_box_count = len(boxes)  # 현재 바운딩 박스 개수

                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    label_id = int(box.cls)
                    color = class_colors.get(label_id, (255, 255, 255))
                    label = class_names.get(label_id, 'Unknown')

                    # 감지된 클래스별 개수 증가
                    if label in detected_counts:
                        detected_counts[label] += 1
                        class_counts[label_id] += 1

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'{label}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

             # freeze_status True일 때 일정 시간 동안 상태를 변경하지 않음
            if freeze_status:
                # print("freeze_status is active, not changing status.")
                time.sleep(100)  # 100초간 상태 고정
                freeze_status = False

            # 감지된 객체가 있는 경우 상태를 True로 변경
            elif any(count > 0 for count in detected_counts.values()):
                with status_lock:
                    status = True
                    # print("Status changed to True due to new detection.")

            # 감지된 객체가 없는 경우 상태를 False로 유지
            else:
                with status_lock:
                    status = False

            # 로그로 상태 출력
            print(f'Status: {status}, Counts: {detected_counts}')

            # 리스트가 비어있다면 최초 감지 시 기록
            if len(box_count_history) == 0 and current_box_count > 0:
                box_count_history.append(current_box_count)
                # send_notification(current_box_count)

            # 바운딩 박스 개수 증가시 리스트 추가, 백엔드 전송
            elif len(box_count_history) > 0 and current_box_count > box_count_history[-1]:
                box_count_history.append(current_box_count)
                current_box.append(current_box_count)
                send_notification(current_box_count)

            # 바운딩 박스 개수 감소시 리스트 추가
            elif len(box_count_history) > 0 and current_box_count < box_count_history[-1]:
                box_count_history.append(current_box_count)

            # 로그로 상태 출력
            print(f'Status: {status}, Counts: {detected_counts},Box Count History: {box_count_history}, back:{current_box}')
            
            with lock:
                cv2.imwrite(img_path, frame)

        time.sleep(sleep_time)


def growth(model, img_path, class_names, class_colors, sleep_time):
    global status, freeze_status
    while True:
        with lock:
            camera.grab()
            ret, frame = camera.retrieve()
        
        if ret:
            results = model(frame)

            # 감지된 객체 처리
            for result in results:
                boxes = result.boxes.xyxy  # (N, 4): 바운딩 박스 좌표 (x1, y1, x2, y2)
                scores = result.boxes.conf  # (N): 각 박스의 신뢰도 점수
                labels = result.boxes.cls   # (N): 각 박스의 클래스 레이블
                
                # Tensor로 변환 (torchvision의 NMS 함수 사용을 위해)
                boxes = boxes.cpu()
                scores = scores.cpu()
                labels = labels.cpu()

                # NMS 수행: 중복된 바운딩 박스 제거
                nms_indices = nms(boxes, scores, iou_threshold=0.2)  # 바운딩박스 겹치는 정도 조정
                
                # NMS 적용된 결과만 사용
                boxes = boxes[nms_indices]
                scores = scores[nms_indices]
                labels = labels[nms_indices]

                # 바운딩 박스 그리기
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i].int().tolist()
                    label_id = int(labels[i])
                    color = class_colors.get(label_id, (255, 255, 255))
                    label = class_names.get(label_id, 'Unknown')

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'{label}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            with lock:
                cv2.imwrite(img_path, frame)

        time.sleep(sleep_time)

# 스레드 생성 (model1에 대한 감지)
thread1 = threading.Thread(target=abnormal, args=(model1, img_path1, class_names_model1, class_colors_model1, 2))
thread1.daemon = True
thread1.start()

# 스레드 생성 (model2에 대한 감지, 나중에 3 >> 0.1로 바꾸기)
thread2 = threading.Thread(target=growth, args=(model2, img_path2, class_names_model2, class_colors_model2, 3))
thread2.daemon = True
thread2.start()

@app.route('/model1')
def index_model1():
    return render_template('index.html')

@app.route('/model2')
def index_model2():
    return render_template('index.html')

@app.route('/image_model1', methods=['GET'])
def get_image_model1():
    with lock:
        if os.path.exists(img_path1):
            with open(img_path1, 'rb') as f:
                image_data = f.read()
            return Response(image_data, mimetype='image/jpeg')
        else:
            return jsonify({'status': 'error', 'message': 'No image available for model1.'})
        
@app.route('/image_model2', methods=['GET'])
def get_image_model2():
    with lock:
        if os.path.exists(img_path2):
            with open(img_path2, 'rb') as f:
                image_data = f.read()
            return Response(image_data, mimetype='image/jpeg')
        else:
            return jsonify({'status': 'error', 'message': 'No image available for model2.'})
        
# 1) 유림 >> 건우 : model1에서 감지된 박스가 있을 때 status를 true로 보내는 API
@app.route('/status', methods=['GET'])
def get_status():
    return jsonify({'status': status})

# 2) 건우 >> 유림 : 버튼 클릭 시 status를 false로 바꾸는 API
@app.route('/reset_status', methods=['POST'])
def reset_status():
    global status, freeze_status
    with status_lock:
        status = False
        freeze_status = True  # 100초간 false로 고정
    print(f'Status = {status}')
    return jsonify({'status': status})

# 3) 유림 >> 건우 : 클래스별 박스 개수를 반환하는 API
@app.route('/get_class_counts', methods=['GET'])
def get_class_counts():
    global class_counts  # 전역 변수 사용
    # 클래스 개수를 JSON 형식으로 반환
    return jsonify({
        'hole': class_counts[0],
        'wither': class_counts[1]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
