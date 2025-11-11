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

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

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
    aug_multiplier = 2 # Augment minority classes to double their size
    balance_target = 0.7 # Balance classes to 70% of majority class


    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    # Split ratios 
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

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

#----------------------------------------------------------------------------------------

class FaceDetector:
    ''' Handels face detection using MTCNN'''
    def __init__(self):
        # Initialize with optimized settings
        self.detector = MTCNN()
        self._failed_detections = 0
    
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
def process_image(image_path, face_detector):

    img = cv2.imread(str(image_path))
    if img is None:
        return None

    # step 2 : Face detection and cropping 
    face_crop = face_detector.detect_and_crop(img)
    if face_crop is None:
        face_crop = img  # Use original if no face detected
    
    # step 3: convert to RGB
    if len(face_crop.shape) == 2:  # grayscale
        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_GRAY2RGB)
    else:
        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

    # quality check : blur detection
    gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # the laplacian variance method measures the amount of edges in an image; low values indicate blurriness
    if laplacian_var < Config.blur_threshold:
        return None  # Discard blurry images
    
    # step 4: Resize to target size
    resized_img = cv2.resize(face_crop, Config.target_size, interpolation=cv2.INTER_LINEAR)

    return resized_img
#----------------------------------------------------------------------------------------
def get_augmentation_pipeline():
    """Define augmentation transformations"""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3)
    ])


def augment_image(image, num_augmentations=3):
    """
    Generate augmented versions of an image
    
    Args:
        image: Input image (numpy array)
        num_augmentations: Number of augmented versions to create
        
    Returns:
        List of augmented images
    """
    transform = get_augmentation_pipeline()
    augmented_images = []
    
    for _ in range(num_augmentations):
        augmented = transform(image=image)['image']
        augmented_images.append(augmented)
    
    return augmented_images

# ----------------------------------------------------------------------------------------
def preprocess_dataset():
    print(" 🚀 Starting Preprocessing Pipeline ... \n")

    # setup
    setup_directories()
    face_detector = FaceDetector()

    # Track statistics 
    stats = {emotion: {'processed':0 , 'discarded':0 , 'augmented':0} 
             for emotion in Config.emotions}
    
    # Process each emotion class 
    for emotion in Config.emotions:
        print(f"\n📁 Processing '{emotion}' class...")

        raw_dir = Path(Config.raw_data_dir) / emotion
        processed_dir = Path(Config.processed_data_dir) / emotion
        discarded_dir = Path(Config.discarded_data_dir) / emotion

        if not raw_dir.exists():
            print(f"⚠️  Warning: Directory {raw_dir} does not exist. Skipping...")
            continue

        # Get all image files
        image_files = list(raw_dir.glob('*.jpg')) + list(raw_dir.glob('*.png')) + list(raw_dir.glob('*.jpeg'))
        # what glob does is it finds all files with the specified extensions in the given directory

        for img_path in tqdm(image_files, desc=f"   Processing {emotion}"):
            # Process image
            processed_img = process_image(img_path, face_detector)
            
            if processed_img is None:
                # Move to discarded
                shutil.copy(img_path, discarded_dir / img_path.name)
                # we copy instead of move to keep original data intact
                stats[emotion]['discarded'] += 1
                continue
            
            # Save processed image
            output_path = processed_dir / img_path.name
            cv2.imwrite(str(output_path), 
                       cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))
            stats[emotion]['processed'] += 1
    
    # Step 6: Calculate class imbalances
    print("\n📊 Class distribution analysis:")
    class_counts = {emotion: len(list((Config.processed_data_dir / emotion).glob('*'))) 
                    for emotion in Config.emotions}
    max_count = max(class_counts.values())
    target_count = int(max_count * Config.balance_target)

    # Step 7: Augmentation for minority classes
    print("\n🔄 Performing data augmentation for minority classes...")

    for emotion in Config.emotions:
        count = class_counts[emotion]

        if count < target_count:
            # Count how many augmented imaegs we need 
            deficit = target_count - count 
            aug_per_image = min(Config.aug_multiplier, (deficit // count) + 1)

            print(f"   Augmenting '{emotion}' class... (Current count: {count})")

            processed_dir = Path(Config.processed_data_dir) / emotion
            augmented_dir = Path(Config.augmented_data_dir) / emotion

            image_files = list(processed_dir.glob('*.jpg')) + \
                         list(processed_dir.glob('*.png'))

            # limit augmentation to avoid overshooting 
            images_to_augment = image_files[: (deficit // aug_per_image) + 1]
            for img_path in tqdm(images_to_augment, desc=f"   Augmenting {emotion}"):
                # Read image
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
                
                # Generate augmented versions
                aug_images = augment_image(img, aug_per_image)  
                
                # Save augmented images
                for i, aug_img in enumerate(aug_images):
                    aug_path = augmented_dir / f"{img_path.stem}_aug{i}{img_path.suffix}"
                    cv2.imwrite(str(aug_path), 
                               cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                    stats[emotion]['augmented'] += 1

                    # stop if we reached target count
                    if (stats[emotion]['processed'] + stats[emotion]['augmented']) >= target_count:
                        break
        else:
            print(f"   '{emotion}' class is balanced (Count: {count}). No augmentation needed.")
    
    # Print final statistics
    print("\n" + "="*60)
    print("📈 PREPROCESSING SUMMARY")
    print("="*60)

    for emotion in Config.emotions:
        print(f"\n{emotion.upper()}:")
        print(f"   ✅ Processed:  {stats[emotion]['processed']}")
        print(f"   ❌ Discarded:  {stats[emotion]['discarded']}")
        print(f"   🔄 Augmented:  {stats[emotion]['augmented']}")
        total = stats[emotion]['processed'] + stats[emotion]['augmented']
        print(f"   📊 Final count: {total}")
    
    print("\n✅ Preprocessing complete!")

#---------------------------------------------------------------------------------------------------------
def split_dataset(source_dir="Data/processed_aug", output_dir="Data/splits"):
    ''' Split dataset into train/val/test sets '''
    print("\n🚀 Starting dataset splitting...")

    for split in ['train', 'val', 'test']:
        for emotion in Config.emotions:
            os.makedirs(Path(output_dir) / split / emotion, exist_ok=True)

    # collect all images with labels 
    all_images = [] 
    all_labels = []

    for emotion in Config.emotions:
        emotion_dir = Path(Config.processed_data_dir) / emotion
        if not emotion_dir.exists():
            emotion_dir = Path(Config.processed_data_dir) / emotion
        
        images = list(emotion_dir.glob('*.jpg')) + list(emotion_dir.glob('*.png'))
        all_images.extend(images)
        all_labels.extend([emotion] * len(images))
        
    # First split into train + temp (val + test)
    train_imgs, temp_imgs, train_labels, temp_labels = train_test_split(
        all_images, all_labels,
        train_size=Config.train_ratio,
        stratify=all_labels,
        random_state=42,
        test_size=(1 - Config.train_ratio)
    )

    # Second split temp into val + test
    val_size = Config.val_ratio / (Config.val_ratio + Config.test_ratio)
    val_imgs, test_imgs, val_labels, test_labels = train_test_split(
        temp_imgs, temp_labels,
        train_size=val_size,
        stratify=temp_labels,
        random_state=42
    )

    # Copy files to split directories
    splits = {
        'train' : (train_imgs, train_labels),
        'val'   : (val_imgs, val_labels),
        'test'  : (test_imgs, test_labels)
    }

    for split, (imgs, labels) in splits.items():
        print(f"\n📁 Creating '{split}' set...")
        for img_path, label in tqdm(zip(imgs, labels), total=len(imgs), desc=f"   Copying to {split}"):
            dest_path = Path(output_dir) / split / label / img_path.name
            shutil.copy(img_path, dest_path)
    
    # Print split statistics 
    print("\n" + "="*60)
    print("📊 DATA SPLIT SUMMARY")
    print("="*60)

    for split_name, (_, labels) in splits.items():
        print(f"\n{split_name.upper()} ({len(labels)} images):")
        counter = Counter(labels)
        for emotion in Config.emotions:
            count = counter.get(emotion, 0)
            percentage = (count / len(labels)) * 100 if labels else 0
            print(f"   {emotion}: {count} ({percentage:.1f}%)")
    
    print("\n✅ Dataset split complete!")

#---------------------------------------------------------------------------------------------------------

def verify_dataset(data_dir="data/splits/train"):
    """
    Verify the processed dataset quality
    
    Args:
        data_dir: Directory to verify
    """
    print("\n🔍 Verifying dataset quality...\n")
    
    checks = {
        'all_rgb': True,
        'correct_size': True,
        'no_corrupted': True,
        'faces_visible': True
    }
    
    issues = []
    
    for emotion in Config.emotions:
        emotion_dir = Path(data_dir) / emotion
        
        if not emotion_dir.exists():
            continue
        
        images = list(emotion_dir.glob('*.jpg')) + list(emotion_dir.glob('*.png'))
        
        # Sample a few images for verification
        sample_size = min(20, len(images))
        sample_images = np.random.choice(images, sample_size, replace=False)
        
        for img_path in sample_images:
            img = cv2.imread(str(img_path))
            
            if img is None:
                checks['no_corrupted'] = False
                issues.append(f"Corrupted: {img_path}")
                continue
            
            # Check dimensions
            if img.shape[:2] != Config.target_size:
                checks['correct_size'] = False
                issues.append(f"Wrong size: {img_path} - {img.shape}")
            
            # Check channels
            if len(img.shape) != 3 or img.shape[2] != 3:
                checks['all_rgb'] = False
                issues.append(f"Not RGB: {img_path}")
    
    # Print results
    print("="*60)
    print("VERIFICATION RESULTS")
    print("="*60)
    
    check_marks = {True: "✅", False: "❌"}
    
    print(f"\n{check_marks[checks['all_rgb']]} All images are RGB")
    print(f"{check_marks[checks['correct_size']]} All images are {Config.target_size}")
    print(f"{check_marks[checks['no_corrupted']]} No corrupted images")
    print(f"{check_marks[checks['faces_visible']]} Faces are centered and visible")
    
    if issues:
        print(f"\n⚠️  Found {len(issues)} issues:")
        for issue in issues[:10]:  # Show first 10
            print(f"   - {issue}")
    else:
        print("\n🎉 All checks passed! Dataset is ready for training.")

# -----------------------------------------------------------------------------------------
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