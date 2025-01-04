import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("Models/kFoldInception_0.keras")
IMG_SIZE = 330
Categories = ["basophil", "eosinophil", "erythroblast", "ig", "lymphocyte", "monocyte", "neutrophil", "platelet"]


#  In order not to make the estimations difficult due to the tonal differences in the incoming pictures,
def to_grayscale_then_rgb(visual):
    visual = tf.image.rgb_to_grayscale(visual)
    visual = tf.image.grayscale_to_rgb(visual)
    return visual


tr_data = ImageDataGenerator(rescale=1.0/255.0, preprocessing_function=to_grayscale_then_rgb)
test_data = tr_data.flow_from_directory(directory="Test", target_size=(IMG_SIZE, IMG_SIZE), batch_size=50, class_mode="sparse")

# Predictions
predicts = model.predict(test_data)
predictions = np.reshape(predicts, (-1, 8))
predictions = np.argmax(predictions, axis=1)
label_index = {v: k for k, v in test_data.class_indices.items()}
predictions = [label_index[p] for p in predictions]

success = 0
label_list = []
global_index = 0 

for _, labels in test_data:
    if global_index >= len(predictions):
        break
    real_labels = labels  
    batch_size = len(real_labels)  

    for i in range(batch_size):
        label_list.append(Categories[int(real_labels[i])])
        if Categories[int(real_labels[i])] == predictions[global_index]:
            success += 1
        
        global_index += 1

print(success/len(predictions))
# https://www.analyticsvidhya.com/blog/2021/06/confusion-matrix-for-multi-class-classification/
cm = confusion_matrix(label_list, predictions, labels=Categories)
cm_df = pd.DataFrame(cm, index=Categories, columns=Categories)
plt.figure(figsize=(12, 12))
sns.heatmap(cm_df/np.sum(cm_df), annot=True, fmt='.2%', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()

def explain_with_shap(image, model):
    background = np.random.rand(10, image.shape[0], image.shape[1], 3).astype(np.float32)

    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(background) 
    return shap_values

