from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, BatchNormalization, Activation
from tensorflow.keras.utils import load_img, img_to_array
import shap
import random 

def to_grayscale_then_rgb(image):
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.grayscale_to_rgb(image)
    return image

def gettingData():
    tr_data = ImageDataGenerator(rescale=1.0/255.0, preprocessing_function=to_grayscale_then_rgb)
    data = tr_data.flow_from_directory(
        directory="C:\\Users\\fatih\\OneDrive\\Masaüstü\\562 Machine\\Proje\\Coding\\Toy Project\\Train",
        seed=123,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=50,
        class_mode="sparse",
        shuffle=True
    )

    images, labels = [], []
    for i in range(4):
        x, y = data[i]
        images.append(x)
        labels.append(y)
    print(f"Data is withdrawn")
    images = np.concatenate(images)
    labels = np.concatenate(labels)
    print(f"Data is ready")
    return images, labels
    
IMG_SIZE = 330
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
Counter = 0
foldAccuracy = []
foldLoss = []
foldValidationAccuracy = []
foldValidationLoss = []
images, labels = gettingData()
for train_indexes, validation_indexes in k_fold.split(images, labels):

    train_images, train_labels = images[train_indexes], labels[train_indexes]
    val_images, val_labels = images[validation_indexes], labels[validation_indexes]

    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(50)
    validation_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(50)

    pre_trained_model = InceptionV3(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
    for layer in pre_trained_model.layers[:209]:
        layer.trainable = False
    layers = tf.keras.layers

    x = pre_trained_model.output

    x = AveragePooling2D(pool_size=(2, 2), strides=2, padding="valid")(x)
    x = AveragePooling2D(pool_size=(2, 2), strides=2, padding="valid")(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(8, activation="softmax")(x)

    model = Model(inputs=pre_trained_model.input, outputs=output)
    print(model.summary())
    initial_learning_rate = 1e-4
    lr_schedule = ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    )
    optimizer = RMSprop(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])
    checkpoint = ModelCheckpoint("CheckPoint/checkpoint" + str(Counter) + ".weights.h5", verbose=1, save_best_only=True, save_weights_only=True, mode='auto', save_freq='epoch')
    early = EarlyStopping(patience=3, verbose=1, mode='auto')
    epochs = 1
    history = model.fit(train_dataset, validation_data=validation_dataset, epochs=epochs, verbose=1, callbacks=[checkpoint, early])

    foldAccuracy.append(history.history["sparse_categorical_accuracy"][-1])
    foldLoss.append(history.history["loss"][-1])
    foldValidationAccuracy.append(history.history["val_sparse_categorical_accuracy"][-1])
    foldValidationLoss.append(history.history["val_loss"][-1])

    print(f"Fold {Counter + 1} Results:")
    print(f"Training Accuracy: {history.history['sparse_categorical_accuracy'][-1]:.4f}")
    print(f"Validation Accuracy: {history.history['val_sparse_categorical_accuracy'][-1]:.4f}")
    
    # Model kaydetme
    model.save_weights(f"Models/kFoldInception_{Counter}.weights.h5")
    model.save(f"Models/kFoldInception_{Counter}.h5")
    Counter += 1

    def preprocess_image(image_path, img_size):
        image = load_img(image_path, target_size=(img_size, img_size))
        image_array = img_to_array(image) / 255.0
        image_array = to_grayscale_then_rgb(image_array)
        return image_array

    def explain_with_shap(model, background, test_image):
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(np.expand_dims(test_image, axis=0))
        return shap_values

    selected_categories = ["basophil", "eosinophil", "lymphocyte"]
    test_dir = "Test"
    sample_images = {}
    for category in selected_categories:
        category_path = os.path.join(test_dir, category)
        sample_files = random.sample(os.listdir(category_path), 3)  # Select 3 random images per category
        sample_images[category] = [os.path.join(category_path, f) for f in sample_files]


    background = train_images[np.random.choice(train_images.shape[0], 100, replace=False)]

    for category, image_paths in sample_images.items():
        for image_path in image_paths:
            image_array = preprocess_image(image_path, IMG_SIZE)
            shap_values = explain_with_shap(model, background, image_array)

            # Görselleştirme
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(image_array)
            plt.title(f"Original Image - {category}")

            plt.subplot(1, 2, 2)
            shap.image_plot([shap_values], np.expand_dims(image_array, axis=0))
            plt.title("SHAP Explanation")
            plt.show()


plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, len(foldLoss)), foldLoss, label="train_loss")
plt.plot(np.arange(0, len(foldValidationLoss)), foldValidationLoss, label="validation_loss")
plt.plot(np.arange(0, len(foldAccuracy)), foldAccuracy, label="train_accuracy")
plt.plot(np.arange(0, len(foldValidationAccuracy)), foldValidationAccuracy, label="validation_accuracy")
plt.title("4-fold cross validation")
plt.ylabel("Loss/Accuracy")
plt.xlabel("Fold")
plt.legend(loc="lower left")
plt.show()

