import json
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Constants
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 100
AUTOTUNE = tf.data.AUTOTUNE

# Enable Mixed Precision
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

def parse_tfrecord(example_proto):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.decode_jpeg(example['image'], channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    # Normalize to [0,1]
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(example['label'], tf.int32)
    return image, label

def get_augmenter():
    """Heavy Data Augmentation Layer for training."""
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(factor=0.11), # ~40 degrees
        layers.RandomZoom(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2)),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomContrast(factor=0.2),
        layers.RandomBrightness(factor=0.2)
    ])

def mixup(images, labels, num_classes, alpha=0.2):
    """Mixup augmentation on a batch."""
    batch_size = tf.shape(images)[0]
    weight = tf.random.gamma([batch_size], alpha)
    weight = weight / (weight + tf.random.gamma([batch_size], alpha))
    weight = tf.cast(weight, tf.float32)
    weight = tf.reshape(weight, (batch_size, 1, 1, 1))

    indices = tf.random.shuffle(tf.range(batch_size))
    mixed_images = images * weight + tf.gather(images, indices) * (1. - weight)
    
    # Needs categorical labels for mixup
    labels_one_hot = tf.one_hot(labels, depth=num_classes)
    weight_label = tf.reshape(weight, (batch_size, 1))
    mixed_labels = labels_one_hot * weight_label + tf.gather(labels_one_hot, indices) * (1. - weight_label)
    
    return mixed_images, mixed_labels

def load_dataset(filenames, num_classes, is_training=False):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=AUTOTUNE)
    if is_training:
        dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.batch(BATCH_SIZE)
    if is_training:
        augmenter = get_augmenter()
        dataset = dataset.map(lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=AUTOTUNE)
        dataset = dataset.map(lambda x, y: mixup(x, y, num_classes), num_parallel_calls=AUTOTUNE)
    else:
        # One-hot encode val/test for categorical crossentropy
        dataset = dataset.map(lambda x, y: (x, tf.one_hot(y, depth=num_classes)), num_parallel_calls=AUTOTUNE)
    return dataset.prefetch(AUTOTUNE)

def conv_block(x, filters, apply_maxpool=True, dropout_rate=0.0):
    shortcut = x
    
    x = layers.Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    
    # Residual Connection matching channels
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), padding='same', kernel_initializer='he_normal')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    
    if apply_maxpool:
        x = layers.MaxPooling2D((2, 2))(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)
    return x

def squeeze_excite_block(x, ratio=16):
    filters = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(filters // ratio, activation='relu', kernel_initializer='he_normal')(se)
    se = layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal')(se)
    se = layers.Reshape((1, 1, filters))(se)
    return layers.Multiply()([x, se])

def build_model(num_classes):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Custom 5-Block CNN with Residuals
    x = conv_block(inputs, 32, apply_maxpool=True, dropout_rate=0.25)
    x = conv_block(x, 64, apply_maxpool=True, dropout_rate=0.25)
    x = conv_block(x, 128, apply_maxpool=True, dropout_rate=0.3)
    x = conv_block(x, 256, apply_maxpool=True, dropout_rate=0.3)
    x = conv_block(x, 512, apply_maxpool=False, dropout_rate=0.0) # Apply SE after block 5
    
    x = squeeze_excite_block(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Classifier Head
    x = layers.Dense(1024, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(512, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.4)(x)
    
    # Due to mixed precision, softmax must be float32
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    
    model = models.Model(inputs, outputs, name="FoodNutrientsCustomCNN")
    return model

def get_lr_schedule(total_epochs, warmup_epochs=5, max_lr=1e-2, min_lr=1e-5):
    """Cosine Annealing with Warmup."""
    def scheduler(epoch, lr):
        if epoch < warmup_epochs:
            return max_lr * ((epoch + 1) / warmup_epochs)
        else:
            # Cosine decay
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
    return scheduler

def build_portion_regression_model(backbone_model):
    """Builds a portion regression head off the trained backbone."""
    backbone = models.Model(inputs=backbone_model.input, outputs=backbone_model.layers[-8].output) # Up to GlobalAvgPool
    backbone.trainable = False # Freeze backbone
    
    x = layers.Dense(256, activation='relu')(backbone.output)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(1, activation='linear', dtype='float32')(x) # Target: grams
    
    model = models.Model(backbone.input, outputs, name="PortionRegression")
    return model

def plot_metrics(history, out_dir):
    # Accuracy
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.legend()
    plt.savefig(out_dir / 'accuracy_curve.png')
    
    # Loss
    plt.figure()
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.legend()
    plt.savefig(out_dir / 'loss_curve.png')

def plot_confusion_matrix(model, test_ds, class_names, out_dir):
    y_true = []
    y_pred = []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.savefig(out_dir / 'confusion_matrix.png')

def main():
    base_dir = Path(__file__).resolve().parent.parent
    data_proc_dir = base_dir / "data" / "processed"
    models_dir = base_dir / "models"
    outputs_dir = base_dir / "outputs"
    logs_dir = base_dir / "logs"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Load class mapping
    label_map_path = data_proc_dir / "label_map.json"
    if not label_map_path.exists():
        print("Label map not found! Run preprocess.py first.")
        # Mock for quick execution check
        class_names = [str(i) for i in range(101)]
    else:
        with open(label_map_path) as f:
            label_map = json.load(f)
            class_names = [label_map[str(i)] for i in range(len(label_map))]
            
    num_classes = len(class_names)
    
    # Check if TFRecords exist
    train_record = data_proc_dir / "train.tfrecord"
    val_record = data_proc_dir / "val.tfrecord"
    test_record = data_proc_dir / "test.tfrecord"
    
    if train_record.exists() and val_record.exists():
        train_ds = load_dataset([str(train_record)], num_classes, is_training=True)
        val_ds = load_dataset([str(val_record)], num_classes, is_training=False)
        test_ds = load_dataset([str(test_record)], num_classes, is_training=False)
    else:
        print("TFRecords not found. Script will initialize model without training to check architecture.")
        train_ds, val_ds, test_ds = None, None, None

    model = build_model(num_classes)
    model.summary()
    
    optimizer = tf.keras.optimizers.SGD(momentum=0.9, nesterov=True)
    
    # Compiling model
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
    )
    
    if train_ds is not None:
        print("Starting training...")
        
        cbs = [
            callbacks.ModelCheckpoint(str(models_dir / 'food_classifier_custom.h5'), save_best_only=True, monitor='val_accuracy'),
            callbacks.EarlyStopping(patience=15, monitor='val_accuracy', restore_best_weights=True),
            callbacks.LearningRateScheduler(get_lr_schedule(EPOCHS)),
            callbacks.TensorBoard(log_dir=str(logs_dir))
        ]
        
        history = model.fit(
            train_ds,
            epochs=EPOCHS,
            validation_data=val_ds,
            callbacks=cbs
        )
        
        print("Training complete. Exporting SavedModel...")
        model.save(str(models_dir / 'food_classifier_saved'))
        
        print("Plotting metrics...")
        plot_metrics(history, outputs_dir)
        
        print("Evaluating on test set...")
        loss, top1, top5 = model.evaluate(test_ds)
        print(f"Test Accuracy: {top1*100:.2f}% | Top-5 Accuracy: {top5*100:.2f}%")
        
        plot_confusion_matrix(model, test_ds, class_names, outputs_dir)
        
        # Portion Regression Mock Train
        print("Initializing Portion Size Regression Model based on Backbone...")
        portion_model = build_portion_regression_model(model)
        portion_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        portion_model.save(str(models_dir / 'portion_estimator.h5'))
        print("Portion regression model exported.")


if __name__ == "__main__":
    main()
