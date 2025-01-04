import os
import random
from tensorflow.keras.utils import load_img, img_to_array
from lime import lime_image
from skimage.segmentation import mark_boundaries
import shap
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def to_grayscale_then_rgb(visual):
    visual = tf.image.rgb_to_grayscale(visual)
    visual = tf.image.grayscale_to_rgb(visual)
    return visual

def preprocess_image(image_path, img_size):
    image = load_img(image_path, target_size=(img_size, img_size))
    image_array = img_to_array(image) / 255.0
    image_array = to_grayscale_then_rgb(image_array)
    return image_array

def explain_with_lime(image, model):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image.numpy().astype('double'),
        model.predict,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )
    temp, mask = explanation.get_image_and_mask(
        label=explanation.top_labels[0],
        positive_only=True,
        num_features=10,
        hide_rest=False
    )
    return mark_boundaries(temp, mask)

def explain_with_shap(image, model):
    background = np.random.rand(10, image.shape[0], image.shape[1], 3).astype(np.float32)

    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(background) 
    return shap_values

model = tf.keras.models.load_model("Models/kFoldInception_0.h5")
IMG_SIZE = 330
selected_categories = ["basophil", "eosinophil", "lymphocyte"]
test_dir = "Test"

sample_images = {}
for category in selected_categories:
    category_path = os.path.join(test_dir, category)
    sample_files = random.sample(os.listdir(category_path), 3)  # Select 3 random images per category
    sample_images[category] = [os.path.join(category_path, f) for f in sample_files]

for category, image_paths in sample_images.items():
    for image_path in image_paths:
        image_array = preprocess_image(image_path, IMG_SIZE)

        # lime_explanation = explain_with_lime(image_array, model)

        shap_values = explain_with_shap(image_array, model)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(image_array)
        plt.title(f"Original Image - {category}")

        # plt.subplot(1, 3, 2)
        # plt.imshow(lime_explanation)
        # plt.title("LIME Explanation")

        plt.subplot(1, 3, 3)
        shap.image_plot(shap_values, [np.expand_dims(image_array, axis=0)])
        plt.title("SHAP Explanation")

        plt.show()