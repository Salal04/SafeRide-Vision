import time
from pathlib import Path
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO
import os;

# --- Deep SORT (nwojke/deep_sort) imports ---
# Make sure `deep_sort` repo folder is in PYTHONPATH or installed as package
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections as gdet  # contains create_box_encoder


# ----------------- User params -----------------
VIDEO_PATH = "vid1.mp4"  # apna video path yahan
OUTPUT_PATH = "output4.mp4"     # optional, processed video save karna ho
YOLO_MODEL = "yolov10m.pt"
REID_WEIGHTS = "deep_sort/networks/mars-small128.pb"
MIRROR_INDICATOR_DETECTOR = "mirror-indicator-yolov10m/weights/best.pt"
MAX_COSINE_DISTANCE = 0.2
NN_BUDGET = 100
MIN_CONFIDENCE = 0.3
N_INIT = 3
MAX_AGE = 30
DRAW_TRAILS = True
TRAIL_LEN = 20
halment_DETECTOR_path = "Halment-Detection\\yolo halmet detector .pt"
# ------------------------------------------------
counter = 0;


def xyxy_to_tlwh(box):
    # box = [x1, y1, x2, y2]
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    return [int(x1), int(y1), int(w), int(h)]


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate intersection area
    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)
    inter_area = inter_width * inter_height
    
    # Calculate union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    
    iou = inter_area / union_area if union_area != 0 else 0
    return iou


def box_inside(inner, outer):
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    return ix1 >= ox1 and iy1 >= oy1 and ix2 <= ox2 and iy2 <= oy2





def expand_box(x1,y1,x2,y2,  w, h, pad_ratio=0.30):
    bw = x2-x1
    bh = y2-y1

    pad_w = int(bw * pad_ratio)
    pad_h = int(bh * pad_ratio)

    x1 = max(0, x1-pad_w)
    y1 = max(0, y1-pad_h)
    x2 = min(w, x2+pad_w)
    y2 = min(h, y2+pad_h)

    return x1,y1,x2,y2


def draw_box(image, coords, label="Box", color=(0,255,0)):
    x1, y1, x2, y2 = map(int, coords)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def run_video(video_path, output_path=None):
    # Load YOLOv8 + Mask/Glasses detector
    yolo = YOLO(YOLO_MODEL)
    mirror_indicator_detector = YOLO(MIRROR_INDICATOR_DETECTOR)
    halment_DETECTOR = YOLO(halment_DETECTOR_path)

    # Init DeepSORT
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", MAX_COSINE_DISTANCE, NN_BUDGET)
    tracker = Tracker(metric, max_iou_distance=0.7, max_age=MAX_AGE, n_init=N_INIT)

    encoder = gdet.create_box_encoder(REID_WEIGHTS, batch_size=1)

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    trails = {}
    frame_idx = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # YOLOv8 person detection
        results = yolo(frame, imgsz=640)
        res = results[0]

        motorcycle_boxes = []
        person_boxes  = []
        if hasattr(res, "boxes") and res.boxes is not None:
            for box in res.boxes:
                cls = int(box.cls[0].item()) if hasattr(box.cls, "__getitem__") else int(box.cls.item())
                if cls != 3:  # Assuming class 3 is motorcycle
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].item()) if hasattr(box.conf, "__getitem__") else float(box.conf.item())
                if conf < MIN_CONFIDENCE:
                    continue
                x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, width, height)
                motorcycle_boxes.append([x1, y1, x2, y2, conf])

        if hasattr(res, "boxes") and res.boxes is not None:
            for box in res.boxes:
                cls = int(box.cls[0].item()) if hasattr(box.cls, "__getitem__") else int(box.cls.item())
                if cls != 0:  # 0 for person in COCO dataset
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].item()) if hasattr(box.conf, "__getitem__") else float(box.conf.item())
                if conf < MIN_CONFIDENCE:
                    continue
                x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, width, height)
                person_boxes.append([x1, y1, x2, y2, conf])

        



        detections = []
        if len(motorcycle_boxes) > 0:
            tlwhs = [xyxy_to_tlwh(p[:4]) for p in motorcycle_boxes]
            features = encoder(frame, tlwhs)
            for i, det in enumerate(motorcycle_boxes):
                detections.append(Detection(tlwhs[i], det[4], features[i]))

        tracker.predict()
        tracker.update(detections)


        
        mirror_results = mirror_indicator_detector(frame)
        Halment_results = halment_DETECTOR(frame)
        results_mirror = mirror_results[0]

        for track in tracker.tracks:
            track.clearCords()

        for bikes in tracker.tracks:
            for person in person_boxes:
                a = compute_iou(person , bikes.to_tlbr())
                print('IoU: ' , a)
                if a > 0.3:
                    bikes.set_Person(True , person);
            

        # Process mirror detections
        if hasattr(results_mirror, "boxes") and results_mirror.boxes is not None:
            for box in results_mirror.boxes:
                cls_id = int(box.cls[0].item()) if hasattr(box.cls, "__getitem__") else int(box.cls.item())
                cls_name = results_mirror.names[cls_id]
                mx1, my1, mx2, my2 = box.xyxy[0].cpu().numpy()
                mx1, my1, mx2, my2 = int(mx1), int(my1), int(mx2), int(my2)
                
                for bikes in tracker.tracks:
                    bbox = bikes.to_tlbr()  
                    x1, y1, x2, y2 = map(int, bbox)
                    status = box_inside((mx1, my1, mx2, my2), (x1, y1, x2, y2))
                    if status and "mirror" in cls_name:
                        bikes.setMirror(cls_name, [mx1, my1, mx2, my2])

    

        print('================= going to Halment state ==============================' , Halment_results)
        halment_res = Halment_results[0]
        halment_res = Halment_results[0]  # first YOLO result
        if hasattr(halment_res, "boxes") and halment_res.boxes is not None:
            print('================= going to Halment stat ==============================')
            for box in halment_res.boxes:
                cls_id = int(box.cls[0].item()) if hasattr(box.cls, "__getitem__") else int(box.cls.item())
                cls_name = halment_res.names[cls_id]  # <--- yahan halment_res.names use karo
                mx1, my1, mx2, my2 = box.xyxy[0].cpu().numpy()
                mx1, my1, mx2, my2 = int(mx1), int(my1), int(mx2), int(my2)

                for bikes in tracker.tracks:
                    bbox = bikes.to_tlbr()  
                    x1, y1, x2, y2 = map(int, bbox)
                    for i in bikes.person:
                        person_box = i[:4] 
                        status_li = box_inside((mx1, my1, mx2, my2),(x1, y1, x2, y2))
                        status_person = box_inside((mx1, my1, mx2, my2), person_box )
                        halment_iou = compute_iou((mx1, my1, mx2, my2), person_box )
                        print('halment_iou', halment_iou, 'status_person:', status_person)
                        if ("With Helmet" in cls_name) and (halment_iou > 0.5 or status_person):
                            print('True')
                            bikes.set_Halment(cls_name, [mx1, my1, mx2, my2])
                        else:
                            print('Condition False')
                        if status_li and "licence" in cls_name:
                            bikes.Set_no_Plate(cls_name, [mx1, my1, mx2, my2])
        else:
            print('No helmet boxes detected in this frame')
            
        print('================= Draw State ==============================')
        for bikes in tracker.tracks:
            bbox = bikes.to_tlbr()  
            x1, y1, x2, y2 = map(int, bbox)
            draw_box(frame, (x1, y1, x2, y2), f'bike: {bikes.track_id} , {bikes.no_of_persons}')
            for halments in bikes.Halment:
                draw_box(frame, halments, f'Halemt: {bikes.track_id}')
            for mirrors in bikes.mirror_Cord:
                draw_box(frame, mirrors, f'Mirror: {bikes.track_id}')
                     

        if writer:
            writer.write(frame)
        cv2.imshow("Video Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_video(VIDEO_PATH, OUTPUT_PATH)