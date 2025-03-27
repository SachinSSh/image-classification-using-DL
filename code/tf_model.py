import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np

# Enable mixed precision (30% less memory usage)
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# --------------------------------------------------
# DATA PIPELINE (MODIFIED)
# --------------------------------------------------
train_df = pd.read_csv('/kaggle/input/ai-vs-human-generated-dataset/train.csv')
test_dir = '/kaggle/input/ai-vs-human-generated-dataset/test_data_v2/'

#Fix column names and paths
train_df = train_df.rename(columns={'file_name': 'image_name'})  # Match your CSV's actual column name
train_df['image_name'] = test_dir + train_df['image_name']  # Full paths

# Enhanced augmentation (focus on subtle artifacts)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,  # Reduced from 20
    width_shift_range=0.05,
    height_shift_range=0.05,
    brightness_range=[0.95, 1.05],  # Critical for AI/human
    channel_shift_range=30,  # Helps detect abnormal color patterns
    horizontal_flip=True,
    zoom_range=0.05,  # Smaller than before
    fill_mode='constant',  # Better than 'nearest' for artifacts
    cval=0.5  # Gray fill for blank areas
)

# Modified ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # Split training data for validation
)

# Create training generator
train_gen = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='image_name',
    y_col='label',
    target_size=(224, 224),
    batch_size=32,
    class_mode='raw',
    subset='training'
)

# Create validation generator
val_gen = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='image_name',
    y_col='label',
    target_size=(224, 224),
    batch_size=32,
    class_mode='raw',
    subset='validation'
)

# --------------------------------------------------
# MODEL ARCHITECTURE (CORE PRESERVED, LAYERS ADDED)
# --------------------------------------------------
def build_model():
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3),
        pooling='avg'
    )
    
    # Freeze 90% of base model
    base_model.trainable = False
    
    model = models.Sequential([
        # Data augmentation as part of the model
        layers.RandomContrast(0.1, input_shape=(224, 224, 3)),
        base_model,
        
        # Added layers (focus on error correction)
        layers.BatchNormalization(),
        layers.Dropout(0.65),  # Increased dropout
        layers.Dense(128, activation='relu', kernel_regularizer='l2'),
        layers.LayerNormalization(),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid', dtype='float32')
    ])
    
    return model

# --------------------------------------------------
# TRAINING STRATEGY (MODIFIED)
# --------------------------------------------------
model = build_model()

# Focal Loss for hard examples
loss_fn = tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, from_logits=False)

# AdamW with decoupled weight decay
optimizer = optimizers.AdamW(
    learning_rate=3e-4,
    weight_decay=1e-4
)

model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=['accuracy']
)

# Progressive unfreezing callback
class Unfreezer(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch == 4:  # Unfreeze 25% layers after 5 epochs
            for layer in model.layers[1].layers[-len(model.layers[1].layers)//4:]:
                layer.trainable = True

# --------------------------------------------------
# TRAINING LOOP (MODIFIED SCHEDULE)
# --------------------------------------------------
train_gen = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=test_dir,
    x_col='image_name',
    y_col='label',
    target_size=(224, 224),
    batch_size=32,  # Reduced for memory
    class_mode='raw'
)

# Modified training loop
history = model.fit(
    train_gen,
    epochs=12,
    validation_data=val_gen,  # Use the validation generator we created
    callbacks=[
        Unfreezer(),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2),
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=4, mode='max')
    ]
)
# --------------------------------------------------
# INFERENCE BOOST (NEW)
# --------------------------------------------------
def TTA_predict(image_path, n_aug=7):
    """Test Time Augmentation for hard samples"""
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = test_datagen.random_transform(x)
    x = test_datagen.standardize(x)
    
    augs = []
    for _ in range(n_aug):
        augs.append(test_datagen.random_transform(x))
    
    batch = np.array(augs)
    preds = model.predict(batch, verbose=0)
    return np.median(preds)  # Median better than mean for outliers

import os
from tqdm import tqdm

# 1. Prepare test image paths
test_dir = '/kaggle/input/ai-vs-human-generated-dataset/test_data_v2/'
test_images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.jpg')]

# 2. Create prediction function with TTA
def predict_with_tta(image_path, model, n_aug=5):
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    x = tf.keras.utils.img_to_array(img)
    x = test_datagen.standardize(x)
    
    # Create augmented versions
    augmented = [test_datagen.random_transform(x) for _ in range(n_aug)]
    batch = np.array(augmented)
    
    # Predict and aggregate
    preds = model.predict(batch, verbose=0)
    return np.mean(preds)  # Use mean for probabilistic approach

# 3. Generate predictions
predictions = []
for img_path in tqdm(test_images):
    prob = predict_with_tta(img_path, model)
    predictions.append({
        'image_name': os.path.basename(img_path),
        'label': float(prob > 0.5)  # Threshold at 0.5
    })

# 4. Create submission DataFrame
submission_df = pd.DataFrame(predictions)
submission_df = submission_df.sort_values('image_name')

# 5. Save to CSV
submission_df.to_csv('submission.csv', index=False)
print("Submission file saved!")
