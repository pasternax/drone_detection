from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import os
import struct
import socket
#import RPi.GPIO as GPIO

def load_model(model_path):
    return YOLO(model_path)

def detect_objects(model, img):
    return model(img, stream=True)

def draw_detection(img, box, conf, cls):
    x1, y1, x2, y2 = box.xyxy[0]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    w, h = x2 - x1, y2 - y1
    cvzone.cornerRect(img, (x1, y1, w, h))
    cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

def calculate_centers(img, box):
    x1, y1, x2, y2 = box.xyxy[0]
    return (x1 + x2) // 2, (y1 + y2) // 2, img.shape[1] // 2, img.shape[0] // 2

def draw_line(img, box_center_x, box_center_y, img_center_x, img_center_y):
    img_center_x = int(img_center_x)
    img_center_y = int(img_center_y)
    box_center_x = int(box_center_x)
    box_center_y = int(box_center_y)

    cv2.line(img, (img_center_x, img_center_y), (box_center_x, box_center_y), (0, 255, 0), 2)

def save_frame(img, output_folder, frame_number):
    output_path = os.path.join(output_folder, f"frame_{frame_number:06d}.jpg")
    cv2.imwrite(output_path, img)

def create_video(output_folder, output_video_path):
    img_array = []
    for filename in sorted(os.listdir(output_folder)):
        if filename.endswith(".jpg"):
            img_path = os.path.join(output_folder, filename)
            img = cv2.imread(img_path)
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

def main():
    model_path = 'best.pt'
    model = load_model(model_path)

    cap = cv2.VideoCapture(0)
    
    output_folder = "output_frames"
    os.makedirs(output_folder, exist_ok=True)
    frame_number = 0

    while True:
        success, img = cap.read()
        if not success:
            break  

        results = detect_objects(model, img)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                draw_detection(img, box, conf, cls)
                
                box_center_x, box_center_y, img_center_x, img_center_y = calculate_centers(img, box)
                draw_line(img, box_center_x, box_center_y, img_center_x, img_center_y)

                dist_x = box_center_x - img_center_x
                dist_y = box_center_y - img_center_y
                print(f"Расстояние по x: {dist_x}, Расстояние по y: {dist_y}")

        #save_frame(img, output_folder, frame_number)
        frame_number += 1

        cv2.imshow("Image", img)
        cv2.waitKey(1)
        #return dist_x, dist_y

    cap.release()
    output_video_path = "output_video.mp4"
    create_video(output_folder, output_video_path)
    print("Запись сохранена.")

if __name__ == "__main__":
    classNames = ["drone"]
    main()
