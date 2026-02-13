import os
import gc
import cv2
import shutil
import datetime
import numpy as np
import pandas as pd
import ipywidgets as widgets
import matplotlib.pyplot as plt
import logging, tensorflow as tf

from datetime import datetime

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from IPython.display import display, clear_output, Markdown
tf.get_logger().setLevel(logging.ERROR)

# # SECTURITY


# SECURITY / PATH CONFIGURATION
BASE_WORKDIR = os.path.abspath("driver_monitoring_workspace")  # Define secure base directory to isolate all project files
os.makedirs(BASE_WORKDIR, exist_ok=True)  # Create the workspace 

# Directories
DATASET_ROOT = os.path.join(BASE_WORKDIR, "dataset") 
RAW_VIDEO_ROOT = os.path.join(BASE_WORKDIR, "raw_videos")  
MODELS_DIR = os.path.join(BASE_WORKDIR, "models")  
REPORTS_DIR = os.path.join(BASE_WORKDIR, "reports")  

for _dir in [DATASET_ROOT, RAW_VIDEO_ROOT, MODELS_DIR, REPORTS_DIR]:
    os.makedirs(_dir, exist_ok=True)  # Ensure all project subdirectories exist inside the workspace

SECURITY_LOG = os.path.join(BASE_WORKDIR, "access.log")  # Log file used to track file access

def log_access(action: str, path: str, role: str):
    timestamp = datetime.datetime.now().isoformat()  # Capture timestamp
    with open(SECURITY_LOG, "a") as log_file:
        log_file.write(
            f"[{timestamp}] role={role} action={action} path={path}\n"  # Record role, action, and path for security auditing
        )

ROLES = {
    "admin": {"read", "write"},  # Admin role has permission to read and modify files
    "user": {"read"}             # User role read-only access
}

def ensure_under_base(path: str, action: str = "read", role: str = "user") -> str:
    if not path:
        log_access(action, BASE_WORKDIR, role)  # Log default access
        return BASE_WORKDIR 

    abs_path = os.path.abspath(path)  # Path to prevent path manipulation
    base = os.path.abspath(BASE_WORKDIR)  # Path for comparison

    if os.path.commonpath([abs_path, base]) != base:
        log_access("DENIED", abs_path, role)  # Log denied access attempts outside the workspace
        raise ValueError(
            f"Unsafe path outside project workspace: {abs_path}\n"
            f"Please select a folder inside: {base}"
        )

    if action not in ROLES.get(role, set()):
        log_access("DENIED", abs_path, role)  # Log denied access due to insufficient role permissions
        raise PermissionError(
            f"Role '{role}' is not permitted to perform '{action}' on {abs_path}"
        )

    log_access(action, abs_path, role)  # Log successful access 
    return abs_path  # Return validated and authorized path

# Controls
TRAIN_SEQUENCE_MODEL = True    # Controls training of the hybrid CNN–RNN sequence model
HYBRID_MODEL = "Hybrid_CNN_RNN_Model.keras"  # Saved architecture for the hybrid temporal model
HYBRID_WEIGHTS = "Hybrid_CNN_RNN_Model.weights.h5"  # Separate weights file for reproducibility

EXPORT_TFLITE = False           
SAVE_REPORTS = False          
EXTRACT_FRAMES = False         

# # BASIC CONFIGURATION


DATASET_ROOT = r"C:\Users\JennyZ\Desktop\GCU\Jupyter Dataset\train_data"  # Path to the local training dataset

IMG_SIZE = (224, 224)   # Target image resolution 
BATCH_SIZE = 32         # Number of samples per training to control memory usage and training efficiency
SEED = 42               # Fixed random seed to ensure reproducibility

print("Dataset root:", DATASET_ROOT) 

# # DATA LOAD


train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_ROOT,                  # Load images from the local dataset directory
    labels="inferred",             # Infer class labels
    label_mode="binary",           # Binary labels for two-class classification
    validation_split=0.2,          # 20% of the data for validation
    subset="training",             # This dataset is the training portion
    seed=SEED,                     # Use a fixed seed to ensure consistent train/validation splits
    image_size=IMG_SIZE,           # Resize all images to a uniform size
    batch_size=BATCH_SIZE,         # Group images into batches for efficient training
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_ROOT,
    labels="inferred",
    label_mode="binary",
    validation_split=0.2,
    subset="validation",           # This dataset is the validation portion
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names
print("Class names:", class_names)

# # DATA CONSTRUCTION


SEQUENCE_LENGTH = 8  # Number of consecutive frames used to represent temporal behavior 

def build_sequence_dataset_from_folders(
    root_dir,                         # Root directory containing class subfolders
    class_names,                      
    sequence_length=SEQUENCE_LENGTH,  # Length of each image sequence
    max_sequences_per_class=None,     # Limit to balance classes and control dataset size
):
    sequences = [] 
    labels = []     

    valid_exts = (".jpg", ".jpeg", ".png", ".bmp") 

    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(root_dir, class_name)  # Path to the current class folder
        image_files = [
            f for f in sorted(os.listdir(class_path))    # Sort files to preserve temporal order
            if f.lower().endswith(valid_exts)            # Filter only valid image files
        ]

        for start in range(0, len(image_files) - sequence_length + 1, sequence_length):
            if max_sequences_per_class and start >= max_sequences_per_class * sequence_length:
                break  # Stop once the maximum number of sequences per class is reached

            seq_imgs = []  # Temporary list to hold frames for a single sequence
            for i in range(sequence_length):
                img_path = os.path.join(class_path, image_files[start + i])  
                img = keras_image.load_img(img_path, target_size=IMG_SIZE)  
                img_arr = keras_image.img_to_array(img)                      
                seq_imgs.append(img_arr)                                      

            sequences.append(np.stack(seq_imgs, axis=0))  # Stack frames into a tensor
            labels.append(class_idx)                      # Assign class label to the sequence

    return np.array(sequences), np.array(labels)  

# # DATA PREPROCESSING


# ==== EXTRACT FRAMES FROM RAW VIDEO ====


RAW_VIDEO_ROOT = r"C:\Users\viane\Documents\GCU\raw_videos"  # Local directory containing raw videos

def extract_frames_from_video(
    video_path,                 # Path to the input video file
    output_dir,                 # Directory where extracted frames will be saved
    class_name,                 # Class label
    every_nth_frame=5,       
    max_frames=None           
):

    cap = cv2.VideoCapture(video_path)  # Open the video using OpenCV
    if not cap.isOpened():
        print(f"[WARN] Could not open video: {video_path}") 
        return                                              

    class_output_dir = os.path.join(output_dir, class_name)  # Create a class output folder
    os.makedirs(class_output_dir, exist_ok=True)            

    base_name = os.path.splitext(os.path.basename(video_path))[0]  
    frame_idx = 0        # Total frames read from the video
    saved_count = 0      # Frames successfully saved

    print(f"[INFO] Extracting from {video_path} -> {class_output_dir}")  

    while True:
        ret, frame = cap.read()  # Read the next frame 
        if not ret:
            break               # Stop when no more frames are available

        if frame_idx % every_nth_frame == 0:
            frame_resized = cv2.resize(frame, IMG_SIZE)  # Resize frame 
            filename = f"{base_name}_frame_{frame_idx:06d}.jpg"  # Generate a ordered frame filename
            out_path = os.path.join(class_output_dir, filename)  # Construct output file path
            cv2.imwrite(out_path, frame_resized)                 # Save the processed frame to disk
            saved_count += 1                                     # Increment saved frame counter

            if max_frames is not None and saved_count >= max_frames:
                break  # Stop extraction once the maximum number of frames is reached

        frame_idx += 1  

    cap.release()  # Release video resources to prevent memory leaks
    print(f"[INFO] Saved {saved_count} frames from {video_path}") 

def extract_frames_from_all_videos(
    raw_video_root=RAW_VIDEO_ROOT,        # Directory containing labeled raw video folders
    output_root=DATASET_ROOT,              # Destination directory for extracted image frames
    every_nth_frame=5,                     # Frame rate to control data volume
    max_frames_per_video=None           
):
    if not os.path.exists(raw_video_root):
        print(f"[WARN] RAW_VIDEO_ROOT not found: {raw_video_root}")  
        return                                                      

    video_exts = (".mp4", ".avi", ".mov", ".mkv") 

    for class_name in os.listdir(raw_video_root):
        class_video_dir = os.path.join(raw_video_root, class_name)  # Path to class video folder
        if not os.path.isdir(class_video_dir):
            continue                                           

        print(f"\n[INFO] Processing class: {class_name}") 
        for fname in os.listdir(class_video_dir):
            if not fname.lower().endswith(video_exts):
                continue                              

            video_path = os.path.join(class_video_dir, fname)  # Construct full path to the video file
            extract_frames_from_video(
                video_path=video_path,                      # Input video to process
                output_dir=output_root,                      # Output directory for extracted frames
                class_name=class_name,                       # Class label for organizing frames
                every_nth_frame=every_nth_frame,             # Apply consistent frame sampling rate
                max_frames=max_frames_per_video,         
            )

# Frame extraction enabled
if EXTRACT_FRAMES:
    print("[INFO] EXTRACT_FRAMES = True, starting video to frame extraction...") 
    extract_frames_from_all_videos(
        raw_video_root=RAW_VIDEO_ROOT,        # Directory containing raw labeled videos
        output_root=DATASET_ROOT,              # Destination for storing extracted image frames
        every_nth_frame=5,                     # Frame rate to control dataset size and redundancy
        max_frames_per_video=None              # No hard limit on extracted frames per video
    )
    print("[INFO] Frame extraction complete.") 
else:
    print("[INFO] EXTRACT_FRAMES = False, skipping video frame extraction.")  

# ==== NORMALIZE BRIGHTNESS PER IMAGE ====


def normalize_dataset_images(
    root_dir,                                   # Root directory containing image folders
    valid_exts=(".jpg", ".jpeg", ".png", ".bmp")
):
    if not os.path.isdir(root_dir):
        print(f"[WARN] Root directory does not exist: {root_dir}")  
        return                                                    

    total_processed = 0  # Counter of normalized images

    for class_name in os.listdir(root_dir):

        if globals().get("STOP_REQUESTED", False):
            print("[STOP] Stop requested — aborting dataset normalization.")  # Allow user to interrupt running processes
            break

        class_path = os.path.join(root_dir, class_name)  # Path to the current class folder
        if not os.path.isdir(class_path):
            continue                                   

        print(f"[INFO] Normalizing images in class folder: {class_name}")  
        for fname in os.listdir(class_path):
            if globals().get("STOP_REQUESTED", False):
                print("[STOP] Stop requested — aborting dataset normalization.")
                break

            if not fname.lower().endswith(valid_exts):
                continue                                 # Ignore unsupported file types

            fpath = os.path.join(class_path, fname)      # Construct full image file path
            img = cv2.imread(fpath)                    
            if img is None:
                continue                                 # Skip corrupted images
                
            norm = cv2.normalize(
                img,                                     # Input image array
                None,                                    # Output array 
                alpha=0,                                 # Minimum normalization value
                beta=255,                                # Maximum normalization value
                norm_type=cv2.NORM_MINMAX                # Apply min–max normalization for consistent intensity scaling
            )

            cv2.imwrite(fpath, norm)                     # Overwrite image with normalized version
            total_processed += 1                       

    print(f"[INFO] Dataset normalization finished/stopped. "
          f"Processed {total_processed} images.")       

# ==== RESIZE ALL DATASET IMAGES ====


def resize_dataset_images(
    root_dir,                                   # Root directory containing image folders
    target_size=IMG_SIZE,                  
    valid_exts=(".jpg", ".jpeg", ".png", ".bmp")
):
    if not os.path.isdir(root_dir):
        print(f"[WARN] Root directory does not exist: {root_dir}")
        return                                                 

    total_processed = 0   # Counter of resized images 

    for class_name in os.listdir(root_dir):
        if globals().get("STOP_REQUESTED", False):
            print("[STOP] Stop requested — aborting dataset resizing.")  # Allow safe interruption running processes
            break

        class_path = os.path.join(root_dir, class_name)  
        if not os.path.isdir(class_path):
            continue                                  

        print(f"[INFO] Resizing images in class folder: {class_name}")  
        for fname in os.listdir(class_path):
            if globals().get("STOP_REQUESTED", False):
                print("[STOP] Stop requested — aborting dataset resizing.")  
                break

            if not fname.lower().endswith(valid_exts):
                continue                                 # Ignore unsupported file types

            fpath = os.path.join(class_path, fname)      # Construct full image file path
            img = cv2.imread(fpath)                 
            if img is None:
                continue                                 # Skip corrupted images

            resized = cv2.resize(img, target_size)       # Resize image to uniform dimensions for model consistency
            cv2.imwrite(fpath, resized)                  # Overwrite image with resized version
            total_processed += 1                 

    print(f"[INFO] Dataset resizing finished/stopped. "
          f"Processed {total_processed} images.")        

# ==== REMOVE BROKEN FRAMES ====


def is_blurry_image(img, threshold=100.0):
    if img is None:
        return True                              # Treat unreadable images as invalid
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale for edge detection
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()     # Compute variance as a focus measure
    return fm < threshold                       

def clean_broken_or_blurry_images(
    root_dir,                                   # Directory containing class-based image folders
    valid_exts=(".jpg", ".jpeg", ".png", ".bmp"), 
    blur_threshold=100.0                        # Threshold below which images are considered blurry
):
    if not os.path.isdir(root_dir):
        print(f"[WARN] Root directory does not exist: {root_dir}")  
        return                                                     

    total_checked = 0  # Counter of evaluated images
    total_removed = 0  # Counter of removed images

    for class_name in os.listdir(root_dir):

        if globals().get("STOP_REQUESTED", False):
            print("[STOP] Stop requested — aborting frame cleaning.")  # Allow user to interrupt running cleaning
            break

        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            continue                                   

        print(f"[INFO] Cleaning class folder: {class_name}") 
        for fname in os.listdir(class_path):

            if globals().get("STOP_REQUESTED", False):
                print("[STOP] Stop requested — aborting frame cleaning.")  
                break

            if not fname.lower().endswith(valid_exts):
                continue                                 # Ignore unsupported file types

            fpath = os.path.join(class_path, fname)      # Construct full image file path
            img = cv2.imread(fpath)          
            total_checked += 1                          

            if img is None:
                try:
                    os.remove(fpath)                     # Remove corrupted or unreadable image file
                    total_removed += 1           
                    print(f"  [REMOVED] Broken image: {fpath}")  
                except OSError as e:
                    print(f"  [WARN] Could not delete {fpath}: {e}") 
                continue

            if is_blurry_image(img, threshold=blur_threshold):
                try:
                    os.remove(fpath)                     # Remove blurry image to improve dataset quality
                    total_removed += 1                  
                    print(f"  [REMOVED] Blurry image: {fpath}") 
                except OSError as e:
                    print(f"  [WARN] Could not delete {fpath}: {e}")  

    print(f"[INFO] Frame cleaning stopped. Checked {total_checked} images, "
          f"removed {total_removed} broken or blurry frames.")  # Summarize cleaning results

# ==== CROP FACE / EYE REGION ====


try:
    FACE_CASCADE = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )  # Load pre-trained Haar cascade for frontal face detection
except Exception as e:
    FACE_CASCADE = None                            # Disable face-based ROI if cascade cannot be loaded
    print("[WARN] Could not load face cascade:", e) 

def crop_face_roi(img, target_size=IMG_SIZE, margin_ratio=0.3):
    if img is None or FACE_CASCADE is None:
        return cv2.resize(img, target_size)        # Fallback to resizing full image if detection is unavailable

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # Convert image to grayscale for face detection
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,                            # Control detection scale for multi-size faces
        minNeighbors=5,                             # Reduce false positives by requiring multiple detections
        minSize=(30, 30)                   
    )

    if len(faces) == 0:
        return cv2.resize(img, target_size)        # Fallback when no face is detected

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])  # Select the largest detected face region

    mh = int(h * margin_ratio)                    
    mw = int(w * margin_ratio)                     
    x1 = max(x - mw, 0)                             
    y1 = max(y - mh, 0)
    x2 = min(x + w + mw, img.shape[1])
    y2 = min(y + h + mh, img.shape[0])

    face_roi = img[y1:y2, x1:x2]                    # Extract the face region of interest
    if face_roi.size == 0:
        return cv2.resize(img, target_size)        # Fallback if ROI extraction fails

    face_resized = cv2.resize(face_roi, target_size)  # Resize ROI to match model dimensions
    return face_resized                     

def apply_face_roi_cropping(
    root_dir,                                   # Directory containing class image folders
    valid_exts=(".jpg", ".jpeg", ".png", ".bmp") 
):
    if not os.path.isdir(root_dir):
        print(f"[WARN] Root directory does not exist: {root_dir}")  
        return                                                 

    if FACE_CASCADE is None:
        print("[WARN] FACE_CASCADE not available. Skipping face cropping.")  
        return

    total_processed = 0  # Counter of processed iamges
    
    for class_name in os.listdir(root_dir):
        if globals().get("STOP_REQUESTED", False):
            print("[STOP] Stop requested — aborting face ROI cropping.")  # Allow safe interruption 
            break

        class_path = os.path.join(root_dir, class_name)  # Path to the current class folder
        if not os.path.isdir(class_path):
            continue                                

        print(f"[INFO] Cropping faces in class folder: {class_name}")   
        for fname in os.listdir(class_path):
            if globals().get("STOP_REQUESTED", False):
                print("[STOP] Stop requested — aborting face ROI cropping.") 
                break

            if not fname.lower().endswith(valid_exts):
                continue                              

            fpath = os.path.join(class_path, fname)      # Construct full image file path
            img = cv2.imread(fpath)                 
            if img is None:
                continue                             

            cropped = crop_face_roi(img, target_size=IMG_SIZE)  # Apply region ofinterest cropping
            if cropped is None:
                continue                                 # Skip if cropping fails

            cv2.imwrite(fpath, cropped)                  # Overwrite image with cropped version
            total_processed += 1                      

    print(f"[INFO] Face ROI cropping stopped. Processed {total_processed} frames.") 

# ==== HISTOGRAM EQUALIZATION ====


def apply_clahe_to_image(img):
    if img is None:
        return None                               # Handle missing images

    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)  # Convert image to YCrCb color space
    y, cr, cb = cv2.split(ycrcb)                    # Split channels to apply contrast enhancement

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # Initialize CLAHE to enhance local contrast
    y_eq = clahe.apply(y)                                        # Apply CLAHE to the luminance channel

    ycrcb_eq = cv2.merge((y_eq, cr, cb))           # Merge enhanced luminance back with chrominance channels
    img_eq = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)  # Convert image back to BGR color space
    return img_eq                         

def apply_hist_eq_clahe_to_dataset(
    root_dir,                                 
    valid_exts=(".jpg", ".jpeg", ".png", ".bmp") 
):
    if not os.path.isdir(root_dir):
        print(f"[WARN] Root directory does not exist: {root_dir}") 
        return                                                     

    total_processed = 0  # Counter of processed iamges

    for class_name in os.listdir(root_dir):
        if globals().get("STOP_REQUESTED", False):
            print("[STOP] Stop requested — aborting histogram equalization.")  # Allow user to interrupt process
            break

        class_path = os.path.join(root_dir, class_name)  # Path to the current class folder
        if not os.path.isdir(class_path):
            continue                               

        print(f"[INFO] Applying CLAHE in class folder: {class_name}")  
        for fname in os.listdir(class_path):
            if globals().get("STOP_REQUESTED", False):
                print("[STOP] Stop requested — aborting histogram equalization.") 
                break

            if not fname.lower().endswith(valid_exts):
                continue                                

            fpath = os.path.join(class_path, fname)      # Construct full image file path
            img = cv2.imread(fpath)               
            if img is None:
                continue                                 

            img_eq = apply_clahe_to_image(img)            # Apply local contrast enhancement to the image
            if img_eq is None:
                continue                      

            cv2.imwrite(fpath, img_eq)                    # Overwrite image with contrast enhanced version
            total_processed += 1                      

    print(f"[INFO] Histogram equalization stopped. "
          f"Processed {total_processed} frames.")         

def export_preprocessed_copies_to_root(
    root_dir,                                   # Directory containing class image folders
    valid_exts=(".jpg", ".jpeg", ".png", ".bmp")  
):
    if not os.path.isdir(root_dir):
        print(f"[WARN] Root directory does not exist: {root_dir}")  
        return                                                    

    total_copied = 0  # Counter of processed iamges

    for class_name in os.listdir(root_dir):
        if globals().get("STOP_REQUESTED", False):
            print("[STOP] Stop requested — aborting flat export.")  # Allow user to interrupt process
            break

        class_path = os.path.join(root_dir, class_name) 
        if not os.path.isdir(class_path):
            continue                                   

        print(f"[INFO] Exporting preprocessed images from class folder: {class_name}")  
        for fname in os.listdir(class_path):
            if globals().get("STOP_REQUESTED", False):
                print("[STOP] Stop requested — aborting flat export.")
                break

            if not fname.lower().endswith(valid_exts):
                continue                            

            src_path = os.path.join(class_path, fname)   # Path of the preprocessed image

            base, ext = os.path.splitext(fname)           # Split filename and extension
            dest_name = f"{class_name}_{base}{ext}"       
            dest_path = os.path.join(root_dir, dest_name) # Destination path in the directory

            counter = 1
            while os.path.exists(dest_path):
                dest_name = f"{class_name}_{base}_{counter}{ext}"  # Ensure filename uniqueness
                dest_path = os.path.join(root_dir, dest_name)
                counter += 1

            shutil.copy2(src_path, dest_path)             # Copy file while preserving metadata
            total_copied += 1                           

    print(f"[INFO] Flat export finished/stopped. Copied {total_copied} images "
          f"into '{root_dir}'.")                         

class_dirs = []  # List to store detected class folder names
for entry in os.listdir(DATASET_ROOT):
    full_path = os.path.join(DATASET_ROOT, entry)  # Construct path for each entry
    if os.path.isdir(full_path):
        class_dirs.append(entry)                   # Collect valid class directories

if not class_dirs:
    print("No class folders found")              
else:
    print("Detected class folders", class_dirs)   

    image_exts = (".jpg", ".jpeg", ".png", ".bmp")  

    class_names_cd = []   # List to store class names
    class_counts_cd = []  # List to store image counts per class

    for class_name in sorted(class_dirs):
        class_path = os.path.join(DATASET_ROOT, class_name)  # Path to the class folder
        count = len([
            f for f in os.listdir(class_path)
            if f.lower().endswith(image_exts)                # Count valid image files
        ])
        class_names_cd.append(class_name)                    # Store class name
        class_counts_cd.append(count)                        # Store corresponding image count
        print(f"Class '{class_name}': {count} images")     

def compute_class_weights_from_dataset(train_ds):
    counts = {0: 0, 1: 0}
    for _, y in train_ds:
        y_np = y.numpy().astype(int).reshape(-1)
        for v in y_np:
            counts[int(v)] += 1

    total = counts[0] + counts[1]
    w0 = total / (2.0 * counts[0]) if counts[0] > 0 else 1.0
    w1 = total / (2.0 * counts[1]) if counts[1] > 0 else 1.0

    print("Class counts:", counts)
    print("Class weights:", {0: w0, 1: w1})
    return {0: w0, 1: w1}

class_weight = compute_class_weights_from_dataset(train_ds)

# # INPUT PIPELINE OPTIMIZATION


# IMPROVE PERFORMANCE WITH PREFETCH
AUTOTUNE = tf.data.AUTOTUNE                                 # Use TensorFlow’s tuning to optimize input performance
train_ds = train_ds.shuffle(1000).prefetch(buffer_size=AUTOTUNE)  # Shuffle training data for better generalization
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)                 # Prefetch validation batches to speed up evaluation

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),  # Augment data by flipping images to improve robustness to left/right variations
        layers.RandomRotation(0.05),      # Add small rotations to viewpoint variation to reduce overfitting
        layers.RandomZoom(0.1),           # Add zoom variation to improve scale robustness for real-world driver frames
    ],
    name="data_augmentation",            
)

preprocess_input = keras.applications.mobilenet_v2.preprocess_input 

# # BUILD MODEL


base_model = keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),     # Define input size to match frame dimensions
    include_top=False,               # Remove the original classifier head so can build a custom classifier
    weights="imagenet",              # Load pretrained weights to leverage transfer learning for better performance
)                                        

# CNN BACKBONE + MOBILENETV2
def build_cnn_backbone():
    base_model.trainable = False     # Freeze pretrained layers to prevent overfitting

    frame_input = keras.Input(shape=IMG_SIZE + (3,), name="frame_input") 
    x = data_augmentation(frame_input)  # Apply augmentation to improve robustness to real-world variation
    x = preprocess_input(x)             # Normalize inputs using MobileNetV2 preprocessing to match pretrained weight expectations
    x = base_model(x, training=False)   # Run frozen backbone to keep BatchNorm layers stable during training
    x = layers.GlobalAveragePooling2D()(x)  

    cnn_backbone = keras.Model(frame_input, x, name="cnn_backbone")  # Create a reusable backbone model for extraction
    return cnn_backbone  

def build_cnn_rnn_model(sequence_length=15):
    # 1) BASE MOBILENETV2
    base_cnn = keras.applications.MobileNetV2(       # Load a pretrained CNN to extract visual features from each frame
        include_top=False,                           # Remove ImageNet classifier head
        weights="imagenet",                          # Use transfer learning to improve performance
        input_shape=IMG_SIZE + (3,)             
    )
    base_cnn.trainable = False                       # Freeze pretrained CNN to reduce overfitting 

    # 2) CNN BACKBONE (frame-level feature extractor)
    cnn_backbone = keras.Sequential([                # Build a reusable feature extractor applied to each frame in the sequence
        layers.Resizing(IMG_SIZE[0], IMG_SIZE[1]),   # Enforce consistent frame dimensions
        layers.Rescaling(1.0 / 127.5, offset=-1.0),  # Normalize to [-1, 1]
        base_cnn,                                    # Pretrained CNN to extract spatial features 
        layers.GlobalAveragePooling2D(),             # Compress spatial feature maps
    ], name="cnn_backbone")                       

    # 3) SEQUENCE INPUT (batch, time, H, W, C)
    seq_inputs = keras.Input(                        # Define sequence input for temporal modeling
        shape=(sequence_length, IMG_SIZE[0], IMG_SIZE[1], 3),  
        name="seq_inputs"                      
    )

    # 4) APPLY CNN TO EACH FRAME
    x = layers.TimeDistributed(cnn_backbone, name="td_cnn")(seq_inputs)  # Extract per-frame features while preserving time order

    # 5) RNN OVER TIME
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=False), name="bilstm")(x)  # Learn temporal patterns across frames
    x = layers.Dropout(0.3, name="dropout")(x)                                            # Regularize to reduce overfitting

    # 6) BINARY OUTPUT
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)  # Output probability for binary classification

    model = keras.Model(seq_inputs, outputs, name="cnn_rnn_hybrid")    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),           # Adam optimizer for stable training
        loss="binary_crossentropy",                               
        metrics=["accuracy"]                                           # Track accuracy to evaluate performance during training/validation
    )
    return model                                               

base_model.trainable = False  # Freeze the pretrained backbone (reduces overfitting, speeds training)

inputs = keras.Input(shape=IMG_SIZE + (3,), name="image_input")  # Define model input for a single frame
x = data_augmentation(inputs)  # Apply oaugmentation to improve robustness to real-world variation
x = preprocess_input(x)        # Normalize inputs using MobileNetV2 to match trained weight expectations
x = base_model(x, training=False)  # Extract deep visual features with the frozen backbone in inference mode for stable BatchNorm behavior
x = layers.GlobalAveragePooling2D()(x) 
x = layers.Dropout(0.3)(x)              
outputs = layers.Dense(1, activation="sigmoid", name="predictions")(x)  # Output probability for binary classification

model = keras.Model(inputs, outputs, name="TrainedModel")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),  # Adam optimizer for stable training 
    loss="binary_crossentropy",                       
    metrics=[
        "accuracy",                                       # Correct rate measure in predictions
        keras.metrics.Precision(name="precision"),        # Precision to track false positives (important for safety-related alerts)
        keras.metrics.Recall(name="recall")               # Recall to track false negatives (critical for missing drowsiness events)
    ],
)

model.summary() 

# # MEMORY-SAFE SEQUENCE PIPELINE


AUTOTUNE = tf.data.AUTOTUNE

def build_sequence_path_dataset(root_dir, class_names, sequence_length=8, max_sequences_per_class=600):
    seq_paths = []
    labels = []

    for class_idx, cname in enumerate(class_names):
        class_dir = os.path.join(root_dir, cname)
        if not os.path.isdir(class_dir):
            continue
            
        files = sorted([
            os.path.join(class_dir, f)
            for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ])

        sequences_added = 0
        for i in range(0, len(files) - sequence_length + 1, sequence_length):
            seq = files[i:i + sequence_length]
            if len(seq) == sequence_length:
                seq_paths.append(seq)
                labels.append(class_idx)
                sequences_added += 1
                if max_sequences_per_class is not None and sequences_added >= max_sequences_per_class:
                    break

    return seq_paths, np.array(labels, dtype=np.int32)

def _load_and_preprocess_frame(path):
    # Read image
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3) 
    img = tf.image.resize(img, [224, 224])
    img = tf.cast(img, tf.float32) / 255.0
    return img

def _load_sequence(seq_paths, label):
    frames = tf.map_fn(_load_and_preprocess_frame, seq_paths, fn_output_signature=tf.float32)
    return frames, label

def make_tf_sequence_dataset(seq_paths, labels, batch_size=4, shuffle=True):
    seq_paths_tensor = tf.constant(seq_paths)
    labels_tensor = tf.constant(labels)

    ds = tf.data.Dataset.from_tensor_slices((seq_paths_tensor, labels_tensor))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(labels), 2000), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(_load_sequence, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds

# # REFINEMENT TRACKER (BEFORE/AFTER)


def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix_hybrid(cm, class_names):
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix - hybrid model")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.show()

# ==== CNN FRAME ====


results_log = []  # list of dict rows for before/after comparison

def evaluate_frame_model(model, val_ds, class_names, threshold=0.5, tag="BASELINE"):
    all_labels = []
    all_preds = []
    all_probs = []

    for batch_images, batch_labels in val_ds:
        probs = model.predict(batch_images, verbose=0).reshape(-1)
        preds = (probs >= threshold).astype(int)

        all_labels.append(batch_labels.numpy().astype(int).reshape(-1))
        all_preds.append(preds.reshape(-1))
        all_probs.append(probs.reshape(-1))

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)

    cm = confusion_matrix(y_true, y_pred)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    row = {
        "tag": tag,
        "threshold": threshold,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "cm": cm
    }
    return row

def log_and_show_eval(row, class_names):
    results_log.append({k: v for k, v in row.items() if k != "cm"}) 
    print(f"\n=== EVALUATION: {row['tag']} (threshold={row['threshold']}) ===")
    print(f"Accuracy : {row['accuracy']:.4f}")
    print(f"Precision: {row['precision']:.4f}")
    print(f"Recall   : {row['recall']:.4f}")
    print(f"F1-score : {row['f1']:.4f}")
    print("\nConfusion Matrix:\n", row["cm"])
    plot_confusion_matrix(row["cm"], class_names)

def plot_before_after_metrics(results_log):
    if len(results_log) < 2:
        print("[INFO] Need at least 2 evaluations to plot comparison.")
        return

    df = pd.DataFrame(results_log)
    df = df.set_index("tag")[["accuracy", "precision", "recall", "f1"]]

    ax = df.plot(kind="bar", figsize=(9, 4))
    plt.title("Performance Before vs After Refinements")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

# ==== HYBRID (CNN + RNN) ====


hybrid_results_log = []  # stores before/after hybrid metrics rows

def evaluate_hybrid_model(seq_model, X_val_seq, y_val_seq, threshold=0.5, tag="HYBRID"):
    probs = seq_model.predict(X_val_seq, verbose=0).reshape(-1)
    preds = (probs >= threshold).astype(int)

    cm = confusion_matrix(y_val_seq, preds)
    acc = accuracy_score(y_val_seq, preds)
    prec = precision_score(y_val_seq, preds, zero_division=0)
    rec = recall_score(y_val_seq, preds, zero_division=0)
    f1 = f1_score(y_val_seq, preds, zero_division=0)

    row = {
        "tag": tag,
        "threshold": threshold,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "cm": cm
    }
    return row

def log_and_show_hybrid_eval(row, class_names):
    hybrid_results_log.append({k: v for k, v in row.items() if k != "cm"})
    print(f"\n=== HYBRID EVAL: {row['tag']} (threshold={row['threshold']}) ===")
    print(f"Accuracy : {row['accuracy']:.4f}")
    print(f"Precision: {row['precision']:.4f}")
    print(f"Recall   : {row['recall']:.4f}")
    print(f"F1-score : {row['f1']:.4f}")
    print("\nConfusion Matrix:\n", row["cm"])
    plot_confusion_matrix_hybrid(row["cm"], class_names)

def plot_hybrid_before_after(hybrid_results_log):
    import pandas as pd
    if len(hybrid_results_log) < 2:
        print("[INFO] Need at least 2 hybrid evaluations to plot before/after.")
        return
    df = pd.DataFrame(hybrid_results_log).set_index("tag")[["accuracy", "precision", "recall", "f1"]]
    ax = df.plot(kind="bar", figsize=(9, 4))
    plt.title("Hybrid Model Performance Before vs After Refinements")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

# BASELINE SNAPSHOT (BEFORE)
baseline_row = evaluate_frame_model(model, val_ds, class_names, threshold=0.5, tag="BEFORE_BASELINE")
log_and_show_eval(baseline_row, class_names)

# # FRAME MODEL: BASELINE TRAINING


print("FRAME MODEL: BASELINE TRAINING")

# Baseline training (frozen backbone)
base_model.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
    ],
)

early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,
    callbacks=[early_stop],
    class_weight=class_weight
)

after_r1 = evaluate_frame_model(model, val_ds, class_names, threshold=0.5, tag="AFTER_R1_CLASS_WEIGHT")
log_and_show_eval(after_r1, class_names)

# # FRAME MODEL: FINE-TUNING


# REFINEMENT 3
print("\n[INFO] Fine-tuning top layers of MobileNetV2 for the FRAME model...")

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
    ],
)

fine_tune_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

history_ft = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,
    callbacks=[fine_tune_stop],
    class_weight=class_weight
)

after_r3 = evaluate_frame_model(model, val_ds, class_names, threshold=0.5, tag="AFTER_R3_FINE_TUNE")
log_and_show_eval(after_r3, class_names)

# # MODEL TRAINING OR LOAD


# ==== HYBRID (CNN + RNN) MODEL TRAINING ====


if TRAIN_SEQUENCE_MODEL:
    print("Hybrid TRAINING ENABLED.")  
    
    # 1) BUILD SEQUENCE DATASET
    X_seq, y_seq = build_sequence_dataset_from_folders(
        DATASET_ROOT,                   # Local dataset directory 
        class_names,                    # Class folder names used to infer labels for sequence learning
        sequence_length=SEQUENCE_LENGTH,  # Number of frames per sequence 
        max_sequences_per_class=600     # Cap sequences per class to control memory usage
    )

    # TRAIN/VALIDATION SPLIT
    X_train_seq, X_val_seq, y_train_seq, y_val_seq = train_test_split(
        X_seq,                          # Sequence inputs
        y_seq,                          # Binary labels aligned to each sequence
        test_size=0.2,                  # 20% for validation to measure generalization
        random_state=SEED,              # Fixed seed for reproducibility
        stratify=y_seq,                 
    )
    
    # 2) BUILD HYBRID CNN–RNN MODEL
    base_model.trainable = False        # Ensure pretrained CNN layers stay frozen 
    seq_model = build_cnn_rnn_model(sequence_length=SEQUENCE_LENGTH)  # Build hybrid model to learn temporal features

    # PHASE 1
    EPOCHS_HEAD_SEQ = 100               # Training EPOCHS for LSTM training 
# ---------------------------------------------------------------


    
    # ==============================
    # HYBRID BASELINE (BEFORE REFINEMENTS)
    # ==============================
    print("\n[BASELINE] Hybrid BEFORE refinements: training head with frozen CNN (no class_weight, no reduce_lr)")

    baseline_es = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history_hybrid_before = seq_model.fit(
        X_train_seq,
        y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=EPOCHS_HEAD_SEQ,  
        callbacks=[baseline_es],
        batch_size=6
    )

    hyb_before_row = evaluate_hybrid_model(seq_model, X_val_seq, y_val_seq, threshold=0.5, tag="HYBRID_BEFORE")
    log_and_show_hybrid_eval(hyb_before_row, class_names)



# ---------------------------------------------------------------
    early_stop_head_seq = EarlyStopping( # Early stopping prevents unnecessary epochs
        monitor="val_loss",           
        patience=5,                    
        restore_best_weights=True,  
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
    
    print("Phase 1 (Hybrid): Training LSTM + classification head with frozen base...")  
    history_head_seq = seq_model.fit(
        X_train_seq,                  
        y_train_seq,            
        validation_data=(X_val_seq, y_val_seq),  # Validation set for monitoring generalization
        epochs=EPOCHS_HEAD_SEQ,         # Max epochs 
        callbacks=[early_stop_head_seq, reduce_lr],# Early stopping callback to reduce overfitting
        batch_size=6,                   # Small batch size to manage memory for 5D sequence tensors
        class_weight=class_weight
    )
    
    # PHASE 2
    print("Phase 2 (Hybrid): Fine-tuning top layers of MobileNetV2...")  # Second pahse to improve performance through transfer learning refinement

    base_model.trainable = True         # Unfreeze the CNN so selected top layers can adapt to the data
    for layer in base_model.layers[:-30]:
        layer.trainable = False         # Keep most layers frozen

    seq_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),  # Lower learning rate for safe fine-tuning of pretrained features
        loss="binary_crossentropy",                           # Binary loss aligns with two-class sequence prediction
        metrics=[
            "accuracy",                                       
            keras.metrics.Precision(name="precision"),        # Precision to see false positives (unnecessary alerts)
            keras.metrics.Recall(name="recall"),              # Recall to see false negatives (missed drowsiness events)
        ],
    )
    print("Hybrid model recompiled for fine-tuning. New summary:")  
    seq_model.summary()                                           

    EPOCHS_FT_SEQ = 100                 # Training EPOCHS for fine-tuning phase

    early_stop_ft_seq = EarlyStopping( # Early stopping prevents unnecessary epochs
        monitor="val_loss",             
        patience=5,                    
        restore_best_weights=True,      
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )

    history_ft_seq = seq_model.fit(
        X_train_seq,                 
        y_train_seq,                  
        validation_data=(X_val_seq, y_val_seq),  # Validation set for monitoring generalization
        epochs=EPOCHS_FT_SEQ,           # Max epochs 
        callbacks=[early_stop_ft_seq, reduce_lr],  # Early stopping to recude overfitting 
        batch_size=6,  # Small batch size for stable sequence training
        class_weight=class_weight
    )

    print("Hybrid Model training complete.")
# ----------------------------------------------------------------------

    
    hyb_after_row = evaluate_hybrid_model(seq_model, X_val_seq, y_val_seq, threshold=0.5, tag="HYBRID_AFTER")
    log_and_show_hybrid_eval(hyb_after_row, class_names)

    
# ----------------------------------------------------------------------
    seq_model.save_weights(HYBRID_WEIGHTS)    # Save learned weights for reproducibility and inference without retraining
    print(f"Hybrid model weights saved to {HYBRID_WEIGHTS}")  

else:
    print("Hybrid TRAINING DISABLED. Rebuilding model and loading weights...")  
    
    base_model.trainable = False   
    seq_model = build_cnn_rnn_model(sequence_length=SEQUENCE_LENGTH)  

    seq_model.load_weights(HYBRID_WEIGHTS) 
    print(f"Loaded hybrid model weights from {HYBRID_WEIGHTS}")  

# # MODEL INTERPRETABILITY


plot_before_after_metrics(results_log)

plot_hybrid_before_after(hybrid_results_log)

# ==== CONFUSION MATRIX ====


print("Evaluating on validation set") 

all_labels = []  
all_preds = []   

for batch_images, batch_labels in val_ds:  # Iterate using validation dataset

    batch_preds_prob = model.predict(batch_images, verbose=0)  # Generate predicted probabilities for each image
    batch_preds = (batch_preds_prob > 0.5).astype(int)         # Convert probabilities to binary using 0.5 threshold

    # Store labels for evaluation
    all_labels.append(batch_labels.numpy().astype(int))        
    all_preds.append(batch_preds)                            

all_labels = np.concatenate(all_labels, axis=0).reshape(-1)   
all_preds = np.concatenate(all_preds, axis=0).reshape(-1)    

# CONFUSION MATRIX
cm = confusion_matrix(all_labels, all_preds)  # Compute confusion matrix to analyze classification errors
print("\nConfusion Matrix:")
print(cm)               

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))  

# Generate precision, recall, and F1-score to evaluate model performance

def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(5, 4))     # Create a figure for visualizing the confusion matrix
    im = ax.imshow(cm, interpolation="nearest")  # Display confusion matrix as an image
    ax.figure.colorbar(im, ax=ax)           
    ax.set(
        xticks=np.arange(len(class_names)),    
        yticks=np.arange(len(class_names)),   
        xticklabels=class_names,               
        yticklabels=class_names,              
        ylabel="True label",                  
        xlabel="Predicted label",             
        title="Confusion Matrix",          
        )
    thresh = cm.max() / 2.0                 
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),         
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",  
            )

    plt.tight_layout()                        
    plt.show()                                  # Display the confusion matrix plot

plot_confusion_matrix(cm, class_names)         

# ==== GRADCAM HEATMAP ====


def load_and_preprocess_image(img_path, img_size=IMG_SIZE):
    img = keras_image.load_img(img_path, target_size=img_size)  # Load image from disk 
    img_arr = keras_image.img_to_array(img)                      # Convert PIL image to NumPy array
    img_arr = np.expand_dims(img_arr, axis=0)                    # Add batch dimension 
    return img, img_arr                             

def make_gradcam_heatmap_tensor(img_array, base_model, dense_layer, preprocess_input_fn):

    w, b = dense_layer.get_weights()                              # Extract weights and bias 

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)  # Convert image array to TensorFlow tensor

    with tf.GradientTape() as tape:
        x = preprocess_input_fn(img_tensor)                      # Apply model preprocessing
        conv_outputs = base_model(x, training=False)             # Pass through CNN backbone to get feature maps
        tape.watch(conv_outputs)                                 # Track gradients with respect to convolutional outputs

        gap = tf.reduce_mean(conv_outputs, axis=[1, 2])         

        logits = tf.matmul(gap, w) + b                      
        probs = tf.math.sigmoid(logits)                           # Convert logits to probabilities
        class_channel = probs[:, 0]                              

    grads = tape.gradient(class_channel, conv_outputs)        
    grads = grads[0]                                              # Remove batch dimension from gradients
    conv_outputs = conv_outputs[0]                                # Remove batch dimension from feature maps

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))             # Average gradients across spatial dimensions

    conv_outputs = conv_outputs * pooled_grads                   

    heatmap = tf.reduce_mean(conv_outputs, axis=-1).numpy()       # Collapse channels to form raw heatmap

    heatmap = np.maximum(heatmap, 0)                              # Apply ReLU to keep only positive contributions
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)                                # Normalize heatmap for visualization

    return heatmap                                               

def display_gradcam(original_img, heatmap, alpha=0.4, save_path=None):
    heatmap_resized = np.uint8(255 * heatmap)                     # Scale heatmap 
    heatmap_resized = np.expand_dims(heatmap_resized, axis=-1)    # Add channel dimension
    heatmap_resized = np.repeat(heatmap_resized, 3, axis=-1)      # Convert heatmap to 3-channel format
    heatmap_resized = tf.image.resize(heatmap_resized, IMG_SIZE).numpy().astype("uint8")  

    img = keras_image.img_to_array(original_img).astype("uint8")  # Convert original image to NumPy array
    img = tf.image.resize(img, IMG_SIZE).numpy().astype("uint8")  # Resize original image for display

    heatmap_color = cv2.applyColorMap(heatmap_resized[..., 0], cv2.COLORMAP_JET)  # Apply color map to heatmap
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)                # Convert color space to RGB

    superimposed_img = heatmap_color * alpha + img               # Overlay heatmap on original image
    superimposed_img = np.clip(superimposed_img, 0, 255).astype("uint8") 

    fig = plt.figure(figsize=(6, 3))                            
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.axis("off")
    plt.imshow(img.astype("uint8"))                               # Display original image

    plt.subplot(1, 2, 2)
    plt.title("Grad-CAM")
    plt.axis("off")
    plt.imshow(superimposed_img)                                  # Display Grad-CAM overlay
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")          

    plt.show()                                                    # Render the visualization

# GRAD-CAM ON THE CNN BACKBONE (MOBILENETV2)
sample_img_path = os.path.join(  # Build a full file path to a sample image for explainability testing
    DATASET_ROOT,            
    "drowsy",                    # Select the "drowsy" class folder as positive-class example
    sorted(os.listdir(os.path.join(DATASET_ROOT, "drowsy")))[0] 
)

original_img, img_array = load_and_preprocess_image(sample_img_path, img_size=IMG_SIZE)  # Create batched array

cnn_backbone_for_gradcam = base_model        # Use the pretrained CNN backbone as the feature map source for Grad-CAM
dense_layer_for_gradcam = model.get_layer("predictions") 

heatmap = make_gradcam_heatmap_tensor(  # Compute Grad-CAM heatmap showing regions that most influenced the prediction
    img_array,                        
    base_model=cnn_backbone_for_gradcam,
    dense_layer=dense_layer_for_gradcam, 
    preprocess_input_fn=preprocess_input,
)

display_gradcam(original_img, heatmap, alpha=0.4)  # Overlay heatmap on the original image for interpretability

# LOAD AND PREPROCESS
original_img, img_array = load_and_preprocess_image(sample_img_path)  # Reload the same sample image for a second Grad-CAM run

dense_layer = model.get_layer("predictions")  # Reference the same prediction layer used to map features to output probability

# COMPUTE HEATMAP
heatmap = make_gradcam_heatmap_tensor(  # Recompute Grad-CAM heatmap using the model backbone and prediction layer
    img_array,                           
    base_model=base_model,             
    dense_layer=dense_layer,           
    preprocess_input_fn=preprocess_input,
)

# DISPLAY
display_gradcam(original_img, heatmap)  # Display the Grad-CAM visualization to support transparent model interpretation

# # TESTER


# SEQUENCE PREDICTION HELPERS FOR CNN-RNN MODEL
def load_sequence_from_folder(folder_path, img_size=IMG_SIZE, sequence_length=16):
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp") 
    all_files = sorted([                            # Sort files to preserve temporal order of frames
        f for f in os.listdir(folder_path)
        if f.lower().endswith(valid_exts)          
    ])

    if not all_files:
        raise ValueError(f"No image files found in folder: {folder_path}")  
        
    selected_files = all_files[:sequence_length]     # Select the first N frames to form a fixed-length input sequence

    frames = []                                      # List to store frame arrays
    file_list = []                                   # List to store file paths used for traceability
    for fname in selected_files:
        fpath = os.path.join(folder_path, fname)     # Construct full path to each frame
        img = keras_image.load_img(fpath, target_size=img_size)  # Load and resize frame to model input size
        img_arr = keras_image.img_to_array(img)  
        frames.append(img_arr)                       # Add frame array to sequence
        file_list.append(fpath)                

    seq_array = np.stack(frames, axis=0)             # Stack frames into shape for temporal modeling
    seq_array = np.expand_dims(seq_array, axis=0)    # Add batch dimension for model input
    return seq_array, file_list                 

def predict_sequence(seq_model, folder_path, class_names, sequence_length=16, threshold=0.5):

    seq_array, file_list = load_sequence_from_folder(  # Load a fixed-length sequence of frames from a folder
        folder_path,                                 
        img_size=IMG_SIZE,                            
        sequence_length=sequence_length,             
    )

    prob_notdrowsy = float(seq_model.predict(seq_array, verbose=0)[0][0])  # Predict probability for the "notdrowsy" class
    prob_drowsy = 1.0 - prob_notdrowsy                                     # Convert to "drowsy" probability for interpretability

    if prob_notdrowsy >= threshold:
        predicted_idx = 1   # Classify as index 1 when probability meets or exceeds threshold
    else:
        predicted_idx = 0   # Otherwise classify as index 0

    predicted_label = class_names[predicted_idx]      # Map predicted index back to the human-readable class label

    print("Sequence folder:", folder_path)           
    print("Frames used (in order):")                 
    for path in file_list:
        print("  -", path)

    print("\nPredicted label:", predicted_label)      # Display predicted class 
    print(f"  P(drowsy)    = {prob_drowsy:.3f}")      # Report drowsy probability for transparency
    print(f"  P(notdrowsy) = {prob_notdrowsy:.3f}")   # Report notdrowsy probability for transparency
    print(f"  Decision threshold = {threshold:.2f}")  # Report threshold used to convert probabilities into a label
    return predicted_label, prob_drowsy, prob_notdrowsy  

def plot_sequence_confidence(prob_drowsy, prob_notdrowsy, save_path=None):
    labels = ["drowsy", "notdrowsy"]              
    probs = [prob_drowsy, prob_notdrowsy]           

    fig = plt.figure(figsize=(4, 4))                  # Create figure for probability bar chart
    bars = plt.bar(labels, probs)                     # Plot predicted probabilities for quick interpretability

    for bar, p in zip(bars, probs):
        height = bar.get_height()                    
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,     
            height + 0.01,                         
            f"{p:.2f}",                               
            ha="center",
            va="bottom",
        )

    plt.ylim(0.0, 1.05)                               
    plt.ylabel("Probability")                        
    plt.title("Sequence-level confidence")           
    plt.tight_layout()                              

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")  

    plt.show()                                        # Display the bar chart

TEST_SEQUENCE_FOLDER = r"C:\Users\JennyZ\Desktop\GCU\Jupyter Dataset\TEST"  # Folder that contain test sequences 

if not os.path.isdir(TEST_SEQUENCE_FOLDER):
    raise ValueError(f"Folder does not exist: {TEST_SEQUENCE_FOLDER}")     # Fail fast if the test folder path is incorrect
else:
    print("✔ Folder found:", TEST_SEQUENCE_FOLDER)                       

sequence_len_for_test = SEQUENCE_LENGTH                                   # Use the same sequence length configuration as the trained model

try:
    seq_model                                                           # Check whether the hybrid model already exists in memory
except NameError:
    print("[INFO] seq_model not found in memory. Rebuilding architecture (CNN+RNN) and loading weights...")  

    # 1) rebuild the same architecture
    seq_model = build_cnn_rnn_model(
        sequence_length=SEQUENCE_LENGTH,                                  # Ensure architecture matches training configuration
        img_size=IMG_SIZE,                                                # Ensure frame size matches preprocessing pipeline
        num_classes=len(class_names)                                      # Ensure output structure matches defined class labels
    )

    # 2) load weights
    seq_model.load_weights("Hybrid_CNN_RNN_Model.weights.h5")             # Load saved weights to run inference
    print("✔ Weights loaded successfully.")                        

# RUN PREDICTION
pred_label, p_drowsy, p_notdrowsy = predict_sequence(                     # Run sequence-level prediction on the selected test folder
    seq_model,                                                           
    TEST_SEQUENCE_FOLDER,                                              
    class_names,                                                        
    sequence_length=sequence_len_for_test,                                
    threshold=0.5                                                       
)

# PLOT
plot_sequence_confidence(p_drowsy, p_notdrowsy)                           # Visualize model confidence for transparency and reporting

# # TENSORFLOW LITE EXPORT


if EXPORT_TFLITE:
    EXPORT_DIR = "export"                              # Output folder
    os.makedirs(EXPORT_DIR, exist_ok=True)             # Create export directory if it does not exist

    def export_to_tflite(keras_model_path, tflite_filename):

        if not os.path.exists(keras_model_path):
            print(f"[WARN] Keras model not found: {keras_model_path}") 
            return                                                     

        print(f"[INFO] Loading Keras model from: {keras_model_path}")   
        model = keras.models.load_model(keras_model_path)              # Load the saved model for conversion

        print("[INFO] Converting to TFLite...")                         
        converter = tf.lite.TFLiteConverter.from_keras_model(model)     # Create a TFLite converter from the model

        converter.target_spec.supported_ops = [                         # Allow a wider set of ops for compatibility
            tf.lite.OpsSet.TFLITE_BUILTINS,                             
            tf.lite.OpsSet.SELECT_TF_OPS                             
        ]

        try:
            converter._experimental_lower_tensor_list_ops = False       # Improve conversion stability for some sequence models
        except AttributeError:
            pass                                                       

        try:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]        # Enable default optimizations 
        except Exception as e:
            print("[WARN] Could not enable optimizations:", e)   

        tflite_model = converter.convert()                              # Run conversion to create a TFLite  model

        tflite_path = os.path.join(EXPORT_DIR, tflite_filename)       
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)                                       # Write TFLite model bytes to disk for deployment use

        size_kb = os.path.getsize(tflite_path) / 1024                   # Compute file size for reporting
        print(f"[INFO] Saved TFLite model to: {tflite_path} ({size_kb:.1f} KB)")  

    export_to_tflite(
        keras_model_path="Hybrid_CNN_RNN_Model.keras",                  # Export the model for sequence deployment
        tflite_filename="Hybrid_CNN_RNN_Model.tflite"                  
    )

    MODEL = "TrainedModel.h5"                                          
    if os.path.exists(MODEL):
        export_to_tflite(
            keras_model_path=MODEL,                                     
            tflite_filename="Frame_CNN_Model.tflite"              
        )
    else:
        print(f"[INFO] Frame-only model not exported because file not found: {MODEL}")  

else:
    print("\n[INFO] EXPORT_TFLITE = False, skipping TFLite export.") 

# # REPORTS SAVE


if SAVE_REPORTS:
    REPORTS_DIR = "reports"                              # Folder for saving evaluation reports
    os.makedirs(REPORTS_DIR, exist_ok=True)              # Create directory if it does not exist

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  

    # 1) SAVE CONFUSION MATRIX
    try:
        def save_confusion_matrix_figure(cm, class_names, filepath):
            fig, ax = plt.subplots(figsize=(5, 4))       # Create a figure for confusion matrix visualization
            im = ax.imshow(cm, interpolation="nearest")  # Plot confusion matrix values as an image
            ax.figure.colorbar(im, ax=ax)                
            ax.set(
                xticks=np.arange(len(class_names)),     
                yticks=np.arange(len(class_names)),      
                xticklabels=class_names,                 
                yticklabels=class_names,                 
                ylabel="True label",                     
                xlabel="Predicted label",                
                title="Confusion Matrix",               
            )

            thresh = cm.max() / 2.0                     
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(
                        j,
                        i,
                        format(cm[i, j], "d"),          
                        ha="center",
                        va="center",
                        color="white" if cm[i, j] > thresh else "black",  
                    )

            plt.tight_layout()                          
            fig.savefig(filepath, bbox_inches="tight")   # Save confusion matrix figure to disk for documentation
            plt.close(fig)                              
            print(f"[INFO] Saved confusion matrix to: {filepath}")  

        cm_path = os.path.join(REPORTS_DIR, f"confusion_matrix_{timestamp}.png")  
        save_confusion_matrix_figure(cm, class_names, cm_path)                  
    except NameError:
        print("[WARN] cm or class_names not defined; run the evaluation/confusion matrix section first.") 

    # 2) SAVE CLASSIFICATION REPORT AS CSV
    try:
        report_dict = classification_report(
            all_labels,                                  # True labels collected from validation evaluation 
            all_preds,                                   # Predicted labels collected from validation evaluation
            target_names=class_names,                   
            output_dict=True                            
        )
        report_df = pd.DataFrame(report_dict).transpose()  # Convert report dictionary CSV table
        csv_path = os.path.join(REPORTS_DIR, f"classification_report_{timestamp}.csv") 
        report_df.to_csv(csv_path, index=True)           
        print(f"[INFO] Saved classification report to: {csv_path}")  
    except NameError:
        print("[WARN] all_labels/all_preds not defined; run evaluation before saving CSV report.")  
        
    # 3) SAVE A GRAD-CAM IMAGE
    try:
        gradcam_path = os.path.join(REPORTS_DIR, f"gradcam_overlay_{timestamp}.png")  
        display_gradcam(original_img, heatmap, alpha=0.4, save_path=gradcam_path)    # Save explainability figure for transparency
        print(f"[INFO] Saved Grad-CAM overlay to: {gradcam_path}")               
        print(f"[INFO] Saved Grad-CAM overlay to: {gradcam_path}")                 
    except NameError:
        print("[WARN] original_img or heatmap not defined; run the Grad-CAM section first.")  

    # 4) SAVE SEQUENCE CONFIDENCE BAR CHART
    try:
        seq_conf_path = os.path.join(REPORTS_DIR, f"sequence_confidence_{timestamp}.png") 
        plot_sequence_confidence(
            prob_drowsy=p_drowsy,                       # Sequence-level probability for drowsy class from prediction output
            prob_notdrowsy=p_notdrowsy,                 # Sequence-level probability for notdrowsy class from prediction output
            save_path=seq_conf_path                     # Save plot image to the reports directory
        )
        print(f"[INFO] Saved sequence confidence plot to: {seq_conf_path}")  
    except NameError:
        print("[WARN] p_drowsy/p_notdrowsy not defined; run predict_sequence() + plot_sequence_confidence() first.")  

else:
    print("\n[INFO] SAVE_REPORTS = False, skipping report export.")