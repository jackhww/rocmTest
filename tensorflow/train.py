import tensorflow as tf
import logging
from tensorflow.keras import layers, models
import argparse
import sys


#Usage Instructions
def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet-50 on CIFAR-10 using TensorFlow and ROCm.")
    parser.add_argument("--log-file", type=str, default="training.log",
                        help="The file path where logs will be saved. Default: training.log")
    return parser.parse_args()

#Logging configuration
def configure_logging(log_file):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    #Redirect stdout and stderr to the log file
    sys.stdout = open(log_file, "a")
    sys.stderr = sys.stdout

def resnet_block(input_tensor, filters, kernel_size=3, stride=1):
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    shortcut = input_tensor

    if stride != 1 or input_tensor.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride)(input_tensor)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def build_resnet50(input_shape=(32, 32, 3), num_classes=10):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(3, strides=2)(x)

    #ResNet blocks
    filters_list = [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3
    strides_list = [1] * 3 + [2] + [1] * 3 + [2] + [1] * 5 + [2]

    for filters, stride in zip(filters_list, strides_list):
        x = resnet_block(x, filters, stride=stride)

    #Output layer
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs=inputs, outputs=outputs)

class FileLoggingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logging.info(f"Epoch {epoch + 1} Logs: {logs}")

def main():
    args = parse_args()
    configure_logging(args.log_file)

    logging.info("Starting ResNet-50 training on CIFAR-10 with TensorFlow and ROCm.")

    #Load dataset
    logging.info("Loading CIFAR-10 dataset...")
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    #Normalize dataset
    train_images, test_images = train_images / 255.0, test_images / 255.0

    #Check if using GPU and that it is available
    physical_devices = tf.config.list_physical_devices('GPU')
    if not physical_devices:
        logging.error("No GPU devices found. Ensure ROCm is properly configured.")
        raise RuntimeError("No GPU devices found.")
    else:
        logging.info(f"Using GPU: {physical_devices}")

    #Build and compile the model
    logging.info("Building ResNet-50 model...")
    model = build_resnet50()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    #Train the model
    logging.info("Starting model training...")
    model.fit(
        train_images,
        train_labels,
        epochs=10,
        batch_size=64,
        validation_data=(test_images, test_labels),
        verbose=2,
        callbacks=[FileLoggingCallback()]
    )
    logging.info("Training completed successfully.")

if __name__ == "__main__":
    main()
