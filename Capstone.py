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
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import image as keras_image

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from IPython.display import display, clear_output, Markdown
tf.get_logger().setLevel(logging.ERROR)

# # SECTURITY


# SECURITY / PATH CONFIGURATION
BASE_WORKDIR = os.path.abspath("driver_monitoring_workspace")
os.makedirs(BASE_WORKDIR, exist_ok=True)

DATASET_ROOT = os.path.join(BASE_WORKDIR, "dataset")
RAW_VIDEO_ROOT = os.path.join(BASE_WORKDIR, "raw_videos")
MODELS_DIR = os.path.join(BASE_WORKDIR, "models")
REPORTS_DIR = os.path.join(BASE_WORKDIR, "reports")

for _dir in [DATASET_ROOT, RAW_VIDEO_ROOT, MODELS_DIR, REPORTS_DIR]:
    os.makedirs(_dir, exist_ok=True)

def ensure_under_base(path: str) -> str:

    if not path:
        return BASE_WORKDIR

    abs_path = os.path.abspath(path)
    base = os.path.abspath(BASE_WORKDIR)

    if os.path.commonpath([abs_path, base]) != base:
        raise ValueError(
            f"Unsafe path outside project workspace: {abs_path}\n"
            f"Please select a folder inside: {base}"
        )
    return abs_path

TRAIN_MODEL = False              
FINE_TUNE = False                
MODEL = "TrainedModel.h5"

TRAIN_SEQUENCE_MODEL = False
HYBRID_MODEL = "Hybrid_CNN_RNN_Model.keras"
HYBRID_WEIGHTS = "Hybrid_CNN_RNN_Model.weights.h5"

EXPORT_TFLITE = False
SAVE_REPORTS = False
EXTRACT_FRAMES = False 

# # BASIC CONFIGURATION


DATASET_ROOT = r"C:\Users\viane\Documents\GCU\train_data"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

print("Dataset root:", DATASET_ROOT)

# # DATA LOAD


train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_ROOT,
    labels="inferred",
    label_mode="binary",          
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_ROOT,
    labels="inferred",
    label_mode="binary",
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names
print("Class names:", class_names)

# # DATA CONSTRUCTION


SEQUENCE_LENGTH = 8 

def build_sequence_dataset_from_folders(
    root_dir,
    class_names,
    sequence_length=SEQUENCE_LENGTH,
    max_sequences_per_class=None,
):
    sequences = []
    labels = []

    valid_exts = (".jpg", ".jpeg", ".png", ".bmp")

    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(root_dir, class_name)
        image_files = [
            f for f in sorted(os.listdir(class_path))
            if f.lower().endswith(valid_exts)
        ]

        for start in range(0, len(image_files) - sequence_length + 1, sequence_length):
            if max_sequences_per_class and start >= max_sequences_per_class * sequence_length:
                break

            seq_imgs = []
            for i in range(sequence_length):
                img_path = os.path.join(class_path, image_files[start + i])
                img = keras_image.load_img(img_path, target_size=IMG_SIZE)
                img_arr = keras_image.img_to_array(img)
                seq_imgs.append(img_arr)

            sequences.append(np.stack(seq_imgs, axis=0))
            labels.append(class_idx)

    return np.array(sequences), np.array(labels)

# # DATA PREPROCESSING


# ==== EXTRACT FRAMES FROM RAW VIDEO ====


RAW_VIDEO_ROOT = r"C:\Users\viane\Documents\GCU\raw_videos" 

def extract_frames_from_video(
    video_path,
    output_dir,
    class_name,
    every_nth_frame=5,
    max_frames=None
):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Could not open video: {video_path}")
        return

    class_output_dir = os.path.join(output_dir, class_name)
    os.makedirs(class_output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    frame_idx = 0
    saved_count = 0

    print(f"[INFO] Extracting from {video_path} -> {class_output_dir}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every_nth_frame == 0:
            frame_resized = cv2.resize(frame, IMG_SIZE)
            filename = f"{base_name}_frame_{frame_idx:06d}.jpg"
            out_path = os.path.join(class_output_dir, filename)
            cv2.imwrite(out_path, frame_resized)
            saved_count += 1

            if max_frames is not None and saved_count >= max_frames:
                break

        frame_idx += 1

    cap.release()
    print(f"[INFO] Saved {saved_count} frames from {video_path}")

def extract_frames_from_all_videos(
    raw_video_root=RAW_VIDEO_ROOT,
    output_root=DATASET_ROOT,
    every_nth_frame=5,
    max_frames_per_video=None
):
    if not os.path.exists(raw_video_root):
        print(f"[WARN] RAW_VIDEO_ROOT not found: {raw_video_root}")
        return

    video_exts = (".mp4", ".avi", ".mov", ".mkv")

    for class_name in os.listdir(raw_video_root):
        class_video_dir = os.path.join(raw_video_root, class_name)
        if not os.path.isdir(class_video_dir):
            continue

        print(f"\n[INFO] Processing class: {class_name}")
        for fname in os.listdir(class_video_dir):
            if not fname.lower().endswith(video_exts):
                continue

            video_path = os.path.join(class_video_dir, fname)
            extract_frames_from_video(
                video_path=video_path,
                output_dir=output_root,
                class_name=class_name,
                every_nth_frame=every_nth_frame,
                max_frames=max_frames_per_video,
            )

if EXTRACT_FRAMES:
    print("[INFO] EXTRACT_FRAMES = True, starting video to frame extraction...")
    extract_frames_from_all_videos(
        raw_video_root=RAW_VIDEO_ROOT,
        output_root=DATASET_ROOT,
        every_nth_frame=5,      
        max_frames_per_video=None  
    )
    print("[INFO] Frame extraction complete.")
else:
    print("[INFO] EXTRACT_FRAMES = False, skipping video frame extraction.")

# ==== NORMALIZE BRIGHTNESS PER IMAGE ====


def normalize_dataset_images(
    root_dir,
    valid_exts=(".jpg", ".jpeg", ".png", ".bmp")
):
    if not os.path.isdir(root_dir):
        print(f"[WARN] Root directory does not exist: {root_dir}")
        return

    total_processed = 0

    for class_name in os.listdir(root_dir):

        if globals().get("STOP_REQUESTED", False):
            print("[STOP] Stop requested — aborting dataset normalization.")
            break

        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        print(f"[INFO] Normalizing images in class folder: {class_name}")
        for fname in os.listdir(class_path):
            if globals().get("STOP_REQUESTED", False):
                print("[STOP] Stop requested — aborting dataset normalization.")
                break

            if not fname.lower().endswith(valid_exts):
                continue

            fpath = os.path.join(class_path, fname)
            img = cv2.imread(fpath)
            if img is None:
                continue
                
            norm = cv2.normalize(
                img,
                None,
                alpha=0,
                beta=255,
                norm_type=cv2.NORM_MINMAX
            )

            cv2.imwrite(fpath, norm)
            total_processed += 1

    print(f"[INFO] Dataset normalization finished/stopped. "
          f"Processed {total_processed} images.")

# ==== RESIZE ALL DATASET IMAGES ====


def resize_dataset_images(
    root_dir,
    target_size=IMG_SIZE,
    valid_exts=(".jpg", ".jpeg", ".png", ".bmp")
):
    if not os.path.isdir(root_dir):
        print(f"[WARN] Root directory does not exist: {root_dir}")
        return

    total_processed = 0

    for class_name in os.listdir(root_dir):
        if globals().get("STOP_REQUESTED", False):
            print("[STOP] Stop requested — aborting dataset resizing.")
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
                continue

            fpath = os.path.join(class_path, fname)
            img = cv2.imread(fpath)
            if img is None:
                continue

            resized = cv2.resize(img, target_size)
            cv2.imwrite(fpath, resized)
            total_processed += 1

    print(f"[INFO] Dataset resizing finished/stopped. "
          f"Processed {total_processed} images.")

# ==== REMOVE BROKEN FRAMES ====


def is_blurry_image(img, threshold=100.0):
    if img is None:
        return True
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < threshold

def clean_broken_or_blurry_images(
    root_dir,
    valid_exts=(".jpg", ".jpeg", ".png", ".bmp"),
    blur_threshold=100.0
):
    if not os.path.isdir(root_dir):
        print(f"[WARN] Root directory does not exist: {root_dir}")
        return

    total_checked = 0
    total_removed = 0

    for class_name in os.listdir(root_dir):

        if globals().get("STOP_REQUESTED", False):
            print("[STOP] Stop requested — aborting frame cleaning.")
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
                continue

            fpath = os.path.join(class_path, fname)
            img = cv2.imread(fpath)
            total_checked += 1

            if img is None:
                try:
                    os.remove(fpath)
                    total_removed += 1
                    print(f"  [REMOVED] Broken image: {fpath}")
                except OSError as e:
                    print(f"  [WARN] Could not delete {fpath}: {e}")
                continue

            if is_blurry_image(img, threshold=blur_threshold):
                try:
                    os.remove(fpath)
                    total_removed += 1
                    print(f"  [REMOVED] Blurry image: {fpath}")
                except OSError as e:
                    print(f"  [WARN] Could not delete {fpath}: {e}")

    print(f"[INFO] Frame cleaning stopped. Checked {total_checked} images, "
          f"removed {total_removed} broken or blurry frames.")

# ==== CROP FACE / EYE REGION ====


try:
    FACE_CASCADE = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
except Exception as e:
    FACE_CASCADE = None
    print("[WARN] Could not load face cascade:", e)

def crop_face_roi(img, target_size=IMG_SIZE, margin_ratio=0.3):
    if img is None or FACE_CASCADE is None:
        return cv2.resize(img, target_size)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        return cv2.resize(img, target_size)

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])

    mh = int(h * margin_ratio)
    mw = int(w * margin_ratio)
    x1 = max(x - mw, 0)
    y1 = max(y - mh, 0)
    x2 = min(x + w + mw, img.shape[1])
    y2 = min(y + h + mh, img.shape[0])

    face_roi = img[y1:y2, x1:x2]
    if face_roi.size == 0:
        return cv2.resize(img, target_size)

    face_resized = cv2.resize(face_roi, target_size)
    return face_resized

def apply_face_roi_cropping(
    root_dir,
    valid_exts=(".jpg", ".jpeg", ".png", ".bmp")
):
    if not os.path.isdir(root_dir):
        print(f"[WARN] Root directory does not exist: {root_dir}")
        return

    if FACE_CASCADE is None:
        print("[WARN] FACE_CASCADE not available. Skipping face cropping.")
        return

    total_processed = 0

    for class_name in os.listdir(root_dir):
        if globals().get("STOP_REQUESTED", False):
            print("[STOP] Stop requested — aborting face ROI cropping.")
            break

        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        print(f"[INFO] Cropping faces in class folder: {class_name}")
        for fname in os.listdir(class_path):
            if globals().get("STOP_REQUESTED", False):
                print("[STOP] Stop requested — aborting face ROI cropping.")
                break

            if not fname.lower().endswith(valid_exts):
                continue

            fpath = os.path.join(class_path, fname)
            img = cv2.imread(fpath)
            if img is None:
                continue

            cropped = crop_face_roi(img, target_size=IMG_SIZE)
            if cropped is None:
                continue

            cv2.imwrite(fpath, cropped)
            total_processed += 1

    print(f"[INFO] Face ROI cropping stopped. Processed {total_processed} frames.")

# ==== HISTOGRAM EQUALIZATION ====


def apply_clahe_to_image(img):
    if img is None:
        return None

    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y_eq = clahe.apply(y)

    ycrcb_eq = cv2.merge((y_eq, cr, cb))
    img_eq = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)
    return img_eq

def apply_hist_eq_clahe_to_dataset(
    root_dir,
    valid_exts=(".jpg", ".jpeg", ".png", ".bmp")
):
    if not os.path.isdir(root_dir):
        print(f"[WARN] Root directory does not exist: {root_dir}")
        return

    total_processed = 0

    for class_name in os.listdir(root_dir):
        if globals().get("STOP_REQUESTED", False):
            print("[STOP] Stop requested — aborting histogram equalization.")
            break

        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        print(f"[INFO] Applying CLAHE in class folder: {class_name}")
        for fname in os.listdir(class_path):
            if globals().get("STOP_REQUESTED", False):
                print("[STOP] Stop requested — aborting histogram equalization.")
                break

            if not fname.lower().endswith(valid_exts):
                continue

            fpath = os.path.join(class_path, fname)
            img = cv2.imread(fpath)
            if img is None:
                continue

            img_eq = apply_clahe_to_image(img)
            if img_eq is None:
                continue

            cv2.imwrite(fpath, img_eq)
            total_processed += 1

    print(f"[INFO] Histogram equalization stopped. "
          f"Processed {total_processed} frames.")

def export_preprocessed_copies_to_root(
    root_dir,
    valid_exts=(".jpg", ".jpeg", ".png", ".bmp")
):
    if not os.path.isdir(root_dir):
        print(f"[WARN] Root directory does not exist: {root_dir}")
        return

    total_copied = 0

    for class_name in os.listdir(root_dir):
        if globals().get("STOP_REQUESTED", False):
            print("[STOP] Stop requested — aborting flat export.")
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

            src_path = os.path.join(class_path, fname)

            base, ext = os.path.splitext(fname)
            dest_name = f"{class_name}_{base}{ext}"
            dest_path = os.path.join(root_dir, dest_name)

            counter = 1
            while os.path.exists(dest_path):
                dest_name = f"{class_name}_{base}_{counter}{ext}"
                dest_path = os.path.join(root_dir, dest_name)
                counter += 1

            shutil.copy2(src_path, dest_path)
            total_copied += 1

    print(f"[INFO] Flat export finished/stopped. Copied {total_copied} images "
          f"into '{root_dir}'.")

class_dirs = []
for entry in os.listdir(DATASET_ROOT):
    full_path = os.path.join(DATASET_ROOT, entry)
    if os.path.isdir(full_path):
        class_dirs.append(entry)

if not class_dirs:
    print("No class folders found")
else:
    print("Detected class folders", class_dirs)

    image_exts = (".jpg", ".jpeg", ".png", ".bmp")

    class_names_cd = []
    class_counts_cd = []

    for class_name in sorted(class_dirs):
        class_path = os.path.join(DATASET_ROOT, class_name)
        count = len([
            f for f in os.listdir(class_path)
            if f.lower().endswith(image_exts)
        ])
        class_names_cd.append(class_name)
        class_counts_cd.append(count)
        print(f"Class '{class_name}': {count} images")

# # INPUT PIPELINE OPTIMIZATION


# IMPROVE PERFORMANCE WITH PREFETCH
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
    ],
    name="data_augmentation",
)

preprocess_input = keras.applications.mobilenet_v2.preprocess_input

# # BUILD MODEL


base_model = keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet",
)

# CNN BACKBONE + MOBILENETV2
def build_cnn_backbone():
    base_model.trainable = False 

    frame_input = keras.Input(shape=IMG_SIZE + (3,), name="frame_input")
    x = data_augmentation(frame_input)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)

    cnn_backbone = keras.Model(frame_input, x, name="cnn_backbone")
    return cnn_backbone

def build_cnn_rnn_model(sequence_length=15):
    # 1) BASE MOBILENETV2
    base_cnn = keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=IMG_SIZE + (3,)
    )
    base_cnn.trainable = False

    # 2) CNN BACKBONE (frame-level feature extractor)
    cnn_backbone = keras.Sequential([
        layers.Resizing(IMG_SIZE[0], IMG_SIZE[1]),
        layers.Rescaling(1.0 / 127.5, offset=-1.0),
        base_cnn,
        layers.GlobalAveragePooling2D(),
    ], name="cnn_backbone")

    # 3) SEQUENCE INPUT (batch, time, H, W, C)
    seq_inputs = keras.Input(
        shape=(sequence_length, IMG_SIZE[0], IMG_SIZE[1], 3),
        name="seq_inputs"
    )

    # 4) APPLY CNN TO EACH FRAME
    x = layers.TimeDistributed(cnn_backbone, name="td_cnn")(seq_inputs)

    # 5) RNN OVER TIME
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=False), name="bilstm")(x)
    x = layers.Dropout(0.3, name="dropout")(x)

    # 6) BINARY OUTPUT
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(seq_inputs, outputs, name="cnn_rnn_hybrid")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

base_model.trainable = False

inputs = keras.Input(shape=IMG_SIZE + (3,), name="image_input")
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid", name="predictions")(x)

model = keras.Model(inputs, outputs, name="TrainedModel")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall")
    ],
)

model.summary()

# # TRAIN OR LOAD MODEL


# ==== CNN MODEL TRAINING ====


# ==== HYBRID (CNN + RNN) MODEL TRAINING ====


if TRAIN_SEQUENCE_MODEL:
    print("Hybrid TRAINING ENABLED.")

    # 1) BUILD SEQUENCE DATASET
    X_seq, y_seq = build_sequence_dataset_from_folders(
        DATASET_ROOT,
        class_names,
        sequence_length=SEQUENCE_LENGTH,
        max_sequences_per_class=800  
    )

    # TRAIN/VALIDATION SPLIT
    X_train_seq, X_val_seq, y_train_seq, y_val_seq = train_test_split(
        X_seq,
        y_seq,
        test_size=0.2,
        random_state=SEED,
        stratify=y_seq,
    )

    print("Train sequences:", X_train_seq.shape, "Val sequences:", X_val_seq.shape)

    # 2) BUILD HYBRID CNN–RNN MODEL
    base_model.trainable = False
    seq_model = build_cnn_rnn_model(sequence_length=SEQUENCE_LENGTH)

    # PHASE 1
    EPOCHS_HEAD_SEQ = 100

    early_stop_head_seq = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    print("Phase 1 (Hybrid): Training LSTM + classification head with frozen base...")
    history_head_seq = seq_model.fit(
        X_train_seq,
        y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=EPOCHS_HEAD_SEQ,
        callbacks=[early_stop_head_seq],
        batch_size=8,
    )

    # PHASE 2
    print("Phase 2 (Hybrid): Fine-tuning top layers of MobileNetV2...")

    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    seq_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    print("Hybrid model recompiled for fine-tuning. New summary:")
    seq_model.summary()

    EPOCHS_FT_SEQ = 100

    early_stop_ft_seq = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    history_ft_seq = seq_model.fit(
        X_train_seq,
        y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=EPOCHS_FT_SEQ,
        callbacks=[early_stop_ft_seq],
        batch_size=8,
    )

    print("Hybrid Model training complete.")

    seq_model.save_weights(HYBRID_WEIGHTS)
    print(f"Hybrid model weights saved to {HYBRID_WEIGHTS}")

else:
    print("Hybrid TRAINING DISABLED. Rebuilding model and loading weights...")
    
    base_model.trainable = False
    seq_model = build_cnn_rnn_model(sequence_length=SEQUENCE_LENGTH)

    seq_model.load_weights(HYBRID_WEIGHTS)
    print(f"Loaded hybrid model weights from {HYBRID_WEIGHTS}")

# # MODEL INTERPRETABILITY


# ==== CONFUSION MATRIX ====


print("Evaluating on validation set")

all_labels = []
all_preds = []

for batch_images, batch_labels in val_ds:

    batch_preds_prob = model.predict(batch_images, verbose=0)
    batch_preds = (batch_preds_prob > 0.5).astype(int)

    all_labels.append(batch_labels.numpy().astype(int))
    all_preds.append(batch_preds)

all_labels = np.concatenate(all_labels, axis=0).reshape(-1)
all_preds = np.concatenate(all_preds, axis=0).reshape(-1)

# CONFUSION MATRIX
cm = confusion_matrix(all_labels, all_preds)
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest")
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
    plt.show()

plot_confusion_matrix(cm, class_names)

# ==== GRADCAM HEATMAP ====


def load_and_preprocess_image(img_path, img_size=IMG_SIZE):
    img = keras_image.load_img(img_path, target_size=img_size)
    img_arr = keras_image.img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0) 
    return img, img_arr
    

def make_gradcam_heatmap_tensor(img_array, base_model, dense_layer, preprocess_input_fn):

    w, b = dense_layer.get_weights() 

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    with tf.GradientTape() as tape:
        x = preprocess_input_fn(img_tensor)
        conv_outputs = base_model(x, training=False)  
        tape.watch(conv_outputs)

        gap = tf.reduce_mean(conv_outputs, axis=[1, 2]) 

        logits = tf.matmul(gap, w) + b 
        probs = tf.math.sigmoid(logits) 
        class_channel = probs[:, 0]   

    grads = tape.gradient(class_channel, conv_outputs)  
    grads = grads[0]          
    conv_outputs = conv_outputs[0] 

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))  

    conv_outputs = conv_outputs * pooled_grads       

    heatmap = tf.reduce_mean(conv_outputs, axis=-1).numpy() 

    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)

    return heatmap

def display_gradcam(original_img, heatmap, alpha=0.4, save_path=None):
    heatmap_resized = np.uint8(255 * heatmap)
    heatmap_resized = np.expand_dims(heatmap_resized, axis=-1)
    heatmap_resized = np.repeat(heatmap_resized, 3, axis=-1)
    heatmap_resized = tf.image.resize(heatmap_resized, IMG_SIZE).numpy().astype("uint8")

    img = keras_image.img_to_array(original_img).astype("uint8")
    img = tf.image.resize(img, IMG_SIZE).numpy().astype("uint8")

    heatmap_color = cv2.applyColorMap(heatmap_resized[..., 0], cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    superimposed_img = heatmap_color * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype("uint8")

    fig = plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.axis("off")
    plt.imshow(img.astype("uint8"))

    plt.subplot(1, 2, 2)
    plt.title("Grad-CAM")
    plt.axis("off")
    plt.imshow(superimposed_img)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    plt.show()

# GRAD-CAM ON THE CNN BACKBONE (MOBILENETV2)
sample_img_path = os.path.join(
    DATASET_ROOT,
    "drowsy",
    sorted(os.listdir(os.path.join(DATASET_ROOT, "drowsy")))[0]
)

original_img, img_array = load_and_preprocess_image(sample_img_path, img_size=IMG_SIZE)

cnn_backbone_for_gradcam = base_model       
dense_layer_for_gradcam = model.get_layer("predictions")  

heatmap = make_gradcam_heatmap_tensor(
    img_array,
    base_model=cnn_backbone_for_gradcam,
    dense_layer=dense_layer_for_gradcam,
    preprocess_input_fn=preprocess_input,
)

display_gradcam(original_img, heatmap, alpha=0.4)

# LOAD AND PREPROCESS
original_img, img_array = load_and_preprocess_image(sample_img_path)

dense_layer = model.get_layer("predictions")

# COMPUTE HEATMAP
heatmap = make_gradcam_heatmap_tensor(
    img_array,
    base_model=base_model,
    dense_layer=dense_layer,
    preprocess_input_fn=preprocess_input,
)

# DISPLAY
display_gradcam(original_img, heatmap)








# # TESTER
# SEQUENCE PREDICTION HELPERS FOR CNN-RNN MODEL
def load_sequence_from_folder(folder_path, img_size=IMG_SIZE, sequence_length=16):
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
    all_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(valid_exts)
    ])

    if not all_files:
        raise ValueError(f"No image files found in folder: {folder_path}")
        
    selected_files = all_files[:sequence_length]

    frames = []
    file_list = []
    for fname in selected_files:
        fpath = os.path.join(folder_path, fname)
        img = keras_image.load_img(fpath, target_size=img_size)
        img_arr = keras_image.img_to_array(img) 
        frames.append(img_arr)
        file_list.append(fpath)

    seq_array = np.stack(frames, axis=0)         
    seq_array = np.expand_dims(seq_array, axis=0)
    return seq_array, file_list

def predict_sequence(seq_model, folder_path, class_names, sequence_length=16, threshold=0.5):

    seq_array, file_list = load_sequence_from_folder(
        folder_path,
        img_size=IMG_SIZE,
        sequence_length=sequence_length,
    )

    prob_notdrowsy = float(seq_model.predict(seq_array, verbose=0)[0][0])
    prob_drowsy = 1.0 - prob_notdrowsy

    if prob_notdrowsy >= threshold:
        predicted_idx = 1  
    else:
        predicted_idx = 0 

    predicted_label = class_names[predicted_idx]

    print("Sequence folder:", folder_path)
    print("Frames used (in order):")
    for path in file_list:
        print("  -", path)

    print("\nPredicted label:", predicted_label)
    print(f"  P(drowsy)    = {prob_drowsy:.3f}")
    print(f"  P(notdrowsy) = {prob_notdrowsy:.3f}")
    print(f"  Decision threshold = {threshold:.2f}")
    return predicted_label, prob_drowsy, prob_notdrowsy

def plot_sequence_confidence(prob_drowsy, prob_notdrowsy, save_path=None):
    labels = ["drowsy", "notdrowsy"]
    probs = [prob_drowsy, prob_notdrowsy]

    fig = plt.figure(figsize=(4, 4))
    bars = plt.bar(labels, probs)

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

    plt.show()

TEST_SEQUENCE_FOLDER = r"C:\Users\JennyZ\Desktop\GCU\Jupyter Dataset\TEST"  

if not os.path.isdir(TEST_SEQUENCE_FOLDER):
    raise ValueError(f"Folder does not exist: {TEST_SEQUENCE_FOLDER}")
else:
    print("✔ Folder found:", TEST_SEQUENCE_FOLDER)

sequence_len_for_test = SEQUENCE_LENGTH

try:
    seq_model
except NameError:
    print("[INFO] seq_model not found in memory. Rebuilding architecture (CNN+RNN) and loading weights...")

    # 1) rebuild the same architecture
    seq_model = build_cnn_rnn_model(
        sequence_length=SEQUENCE_LENGTH,
        img_size=IMG_SIZE,
        num_classes=len(class_names)
    )

    # 2) load weights (recommended)
    seq_model.load_weights("Hybrid_CNN_RNN_Model.weights.h5")
    print("✔ Weights loaded successfully.")

# RUN PREDICTION
pred_label, p_drowsy, p_notdrowsy = predict_sequence(
    seq_model,
    TEST_SEQUENCE_FOLDER,
    class_names,
    sequence_length=sequence_len_for_test,
    threshold=0.5
)

# PLOT
plot_sequence_confidence(p_drowsy, p_notdrowsy)
