if __name__ == '__main__':
    from ultralytics import YOLO

    # Load the model.
    model = YOLO('yolov8n.pt')
 
    # Training.
    results = model.train(
       data='yolov8_3.yaml',
       imgsz=640,
       epochs=10,
       batch=8,
       name='yolov8_3')
