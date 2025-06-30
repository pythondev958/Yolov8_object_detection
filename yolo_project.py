import argparse
import os
import glob
import urllib.request
from ultralytics import YOLO

def download_image(url, save_path):
    if not os.path.exists(save_path):
        print(f"Downloading sample image to {save_path}...")
        urllib.request.urlretrieve(url, save_path)
        print("Download complete.")
    else:
        print(f"Image already exists at {save_path}.")

def is_image_file(filename):
    return filename.lower().endswith(('.jpg', '.jpeg', '.png'))

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Object Detection")
    parser.add_argument("--source", type=str, default="images", help="Path to input image or folder")
    parser.add_argument("--output", type=str, default="outputs", help="Folder to save output images")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # If source is a folder, get list of image files
    if os.path.isdir(args.source):
        image_files = [f for f in glob.glob(os.path.join(args.source, '*')) if is_image_file(f)]
        if not image_files:
            # No images found: download sample image
            sample_path = os.path.join(args.source, "sample.jpg")
            sample_url = "https://ultralytics.com/images/bus.jpg"
            download_image(sample_url, sample_path)
            image_files = [sample_path]
    else:
        # Source is a file
        if not os.path.exists(args.source) and is_image_file(args.source):
            sample_url = "https://ultralytics.com/images/bus.jpg"
            download_image(sample_url, args.source)
        image_files = [args.source]

    # Load model once
    model = YOLO("yolov8n.pt")

    # Run detection on each image and save output
    for i, image_path in enumerate(image_files):
        print(f"Processing {image_path}...")
        results = model(image_path)
        for j, result in enumerate(results):
            save_path = os.path.join(args.output, f"result_{i}_{j}.jpg")
            result.save(save_path)
            print(f"Saved detection result to: {save_path}")

if __name__ == "__main__":
    main()
