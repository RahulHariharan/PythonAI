#!/usr/bin/env python3
"""
People Counter - Counts the number of people in an image
Requires: ultralytics, opencv-python, pillow
Install with: pip install ultralytics opencv-python pillow
"""

import cv2
from ultralytics import YOLO
from pathlib import Path
import argparse


def count_people(image_path, confidence_threshold=0.25, show_visualization=True):
    """
    Count the number of people in an image using YOLOv8
    
    Args:
        image_path (str): Path to the input image file
        confidence_threshold (float): Minimum confidence for detection (0-1)
        show_visualization (bool): Whether to display annotated image
    
    Returns:
        int: Number of people detected
    """
    # Check if image exists
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load YOLOv8 model (downloads automatically on first run)
    print("Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt')  # 'n' for nano (fastest), can use 's', 'm', 'l', 'x' for better accuracy
    
    # Read the image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    print(f"Processing image: {image_path}")
    print(f"Image size: {img.shape[1]}x{img.shape[0]} pixels")
    
    # Run inference
    results = model(img, conf=confidence_threshold, verbose=False)
    
    # Count people (class 0 in COCO dataset is 'person')
    people_count = 0
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Class 0 is 'person' in COCO dataset
            if cls == 0:
                people_count += 1
    
    print(f"\n{'='*50}")
    print(f"People detected: {people_count}")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"{'='*50}\n")
    
    # Visualize results
    if show_visualization:
        # Annotate image with detections
        annotated_img = results[0].plot()
        
        # Add count text to image
        text = f"People Count: {people_count}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        
        # Draw background rectangle
        cv2.rectangle(
            annotated_img,
            (10, 10),
            (20 + text_width, 30 + text_height),
            (0, 0, 0),
            -1
        )
        
        # Draw text
        cv2.putText(
            annotated_img,
            text,
            (15, 25 + text_height),
            font,
            font_scale,
            (0, 255, 0),
            thickness
        )
        
        # Save annotated image
        output_path = img_path.parent / f"{img_path.stem}_annotated.png"
        cv2.imwrite(str(output_path), annotated_img)
        print(f"Annotated image saved to: {output_path}")
        
        # Display image (resize if too large)
        display_img = annotated_img.copy()
        max_display_size = 1200
        h, w = display_img.shape[:2]
        
        if max(h, w) > max_display_size:
            scale = max_display_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            display_img = cv2.resize(display_img, (new_w, new_h))
        
        cv2.imshow('People Detection', display_img)
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return people_count


def main():
    parser = argparse.ArgumentParser(
        description='Count people in an image using YOLOv8'
    )
    parser.add_argument(
        'image_path',
        type=str,
        help='Path to the input image (PNG format)'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.25,
        help='Confidence threshold for detection (0-1, default: 0.25)'
    )
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Disable visualization of results'
    )
    
    args = parser.parse_args()
    
    try:
        count = count_people(
            args.image_path,
            confidence_threshold=args.confidence,
            show_visualization=not args.no_viz
        )
        print(f"\nFinal count: {count} people")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())