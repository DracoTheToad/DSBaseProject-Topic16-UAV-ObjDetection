from ultralytics import YOLO
from pathlib import Path


PROJECT_NAME = Path('baseline_yolov8')

RUN_DIR = Path('runs/detect')
PROJECT_PATH = RUN_DIR / PROJECT_NAME


def train_from_scratch():
    model = YOLO('yolov8n.pt')
    metrics = model.train(
        data='VisDrone.yaml',
        epochs=100,
        imgsz=640,
        batch=1,
        seed=0,
        device='cuda',
        name=PROJECT_NAME,
        max_det=902,
    )


if PROJECT_PATH.exists():
    continue_ = input('project already exists. 1 to rerun, 0 to start anew: ') == '1'

    if continue_:
        last_weight = PROJECT_PATH / 'weights/last.pt'
        model = YOLO(last_weight)
        model.train(resume=True)
    else: train_from_scratch()

else:
    train_from_scratch()
