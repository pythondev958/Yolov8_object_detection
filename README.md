# YOLOv8 Object Detection

This project performs object detection on images using Ultralytics YOLOv8.

## Features

- Supports batch detection on all images in a folder.
- Automatically downloads a sample image if none is provided.
- Saves detection results with bounding boxes and labels.
- Easy to extend for videos or live camera feed.

## Setup

1. Clone the repository:

```bash
git clone https://github.com/your-username/YOLOv8-Object-Detection.git
cd YOLOv8-Object-Detection

python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt

python yolo_project.py --source images --output outputs/


## Future Improvements

Add video or webcam input support

Build a web interface with Streamlit or Gradio

Train custom YOLO models on your own datasets

