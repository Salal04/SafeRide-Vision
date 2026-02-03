# SafeRide Vision

**SafeRide Vision** is a computer vision project for **motorcycle safety monitoring**.  
It automatically detects whether a rider is wearing a **helmet**, checks for **bike mirrors**, and identifies the **number plate** from traffic camera videos.

The system uses **deep learning-based object detection** (like YOLO and DeepSort) to detect these objects accurately.  
This helps improve **road safety**, **traffic rule enforcement**, and **smart traffic monitoring systems**.

---
## Demo 

![SafeRide Vision Demo](output4-ezgif.com-video-to-gif-converter.gif)

## Features

- Helmet detection for riders  
- Side mirror detection  
- Number plate detection  
- Works with traffic camera video feeds  
- Real-time or recorded video analysis  

---

## Requirements

- Python 3.10+  
- OpenCV  
- PyTorch  
- Ultralytics YOLO  
- Numpy  
- Matplotlib
- DeepSort  

- *(You can install all dependencies using `pip install -r requirements.txt`)*
---

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/Salal04/SafeRide-Vision.git
   
2. Clone The Deepsort:
   ```bash
   git clone "https://github.com/nwojke/deep_sort.git"
   
3. Replace Track File with the given updated Track


## Limitations
1. detect rear but, Unable to detect front no plate due to unavailabilty of dataset
