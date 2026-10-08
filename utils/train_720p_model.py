from ultralytics import YOLO
from pathlib import Path


PROJECT_NAME = Path('720p-image')

RUN_DIR = Path('runs/detect')
PROJECT_PATH = RUN_DIR / PROJECT_NAME


def train_from_scratch():
    model = YOLO('yolo26n.pt')
    metrics = model.train(
        data='VisDrone.yaml',
        epochs=50,
        imgsz=1280,
        batch=2,
        workers=4,
        device='cuda',
        seed=0,
        name=PROJECT_NAME
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
