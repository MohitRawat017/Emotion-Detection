"""
Emotion Detection - Complete Data Preprocessing Pipeline
---------------------------------------------------------------->

Steps :
- Face detection and cropping
- Channel normalization (grayscale → RGB)
- Resizing to standard dimensions
- Quality filtering
- Augmentation for minority classes
- Train/val/test splitting
"""


import os 
import cv2 
import numpy as np 
import shutil 
from pathlib import Path 
from mtcnn import MTCNN # For face detection
from sklearn.model_selection import train_test_split
import albumentations as A # For data augmentation
from collections import Counter # For class distribution analysis
from tqdm import tqdm # For progress bars

# -------------------------------------------------------------------------------------
class Config:
    ''' Central Configuration for preprocessing '''
    raw_data_dir = Path("Data/raw/")
    processed_data_dir = Path("Data/processed/")
    augmented_data_dir = Path("Data/augmented/")
    discarded_data_dir = Path("Data/discarded/")

    # Preprocessing parameters
    target_size = (224, 224) # good for transfer learning
    face_margin = 0.25 # 25% margin around detected face
    blur_threshold = 100.0 

    # Augmentation settings
    aug_multiplier = 3 # Augment minority classes to triple their size

    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    # Split ratios 
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

#----------------------------------------------------------------------------------------

class FaceDetector:
    ''' Handels face detection using MTCNN'''
    def __init__(self):
        self.detector = MTCNN()
    
    def detect_and_crop(self, image):
        
        result = self.detector.detect_faces(image)
        if not result:
            return None  # No face detected
        
        # Assume the first detected face is the primary one
        face = max(result, key=lambda x: x['box'][2] * x['box'][3])
        
        x, y, w, h = face['box']

        # Add margin 
        margin_x = int(w * Config.face_margin)
        margin_y = int(h * Config.face_margin)

        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(image.shape[1], x + w + margin_x)
        y2 = min(image.shape[0], y + h + margin_y)

        # Crop face region 
        cropped_face = image[y1:y2, x1:x2]
        return cropped_face

#----------------------------------------------------------------------------------------
def setup_directories():
    ''' Create necessary directories '''
    dirs = [
        Config.processed_data_dir, 
        Config.augmented_data_dir, 
        Config.discarded_data_dir
    ]

    for emotion in Config.emotions:
        for base_dir in dirs:
            (Path(base_dir) / emotion).mkdir(parents=True, exist_ok=True)
    print("✅ Directory structure created successfully.")

def preprocess_dataset():
    print(" 🚀 Starting Preprocessing Pipeline ... \n")

    # setup
    setup_directories()
    face_detector = FaceDetector()


#---------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    # Run complete pipeline 
    preprocess_dataset()
    split_dataset()
    verify_dataset()

    print("\n" + "="*60)
    print("🎉 ALL PREPROCESSING STEPS COMPLETED!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review the splits in 'data/splits/'")
    print("2. Check discarded images in 'data/discarded/'")
    print("3. Proceed to model training with clean dataset")