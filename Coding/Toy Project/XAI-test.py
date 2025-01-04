import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
import shap
from tensorflow.keras.models import load_model
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

train_path = "C:\\Users\\fatih\\OneDrive\\Masaüstü\\562 Machine\\Proje\\Coding\\Toy Project\\Train"
test_path = "C:\\Users\\fatih\\OneDrive\\Masaüstü\\562 Machine\\Proje\\Coding\\Toy Project\\Explaining-Test"

image_size = (330, 330)

test_datagen = ImageDataGenerator(rescale=1.0/255.0)
train_generator = test_datagen.flow_from_directory(
    train_path,
    target_size=image_size,
    batch_size=100,
    class_mode='categorical',
    shuffle=True
)
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=image_size,
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

simple_model = load_model('best_simple_model.keras')
advanced_model = load_model('best_medium_model.keras')
inception_model = load_model('best_advanced_model.keras')

shap_times = []

for model_name, model in zip(['Advanced Model', 'Simple Model', 'InceptionV3 Model'], [advanced_model, simple_model, inception_model]):
    print(f"{model_name} SHAP ")

    def f(X):
        tmp = X.copy()
        return model(tmp)
    
    masker = shap.maskers.Image("inpaint_telea", train_generator[0][0][0].shape)
    explainer = shap.Explainer(f, masker)
    background = next(train_generator)[0][:15]
    start_time = time.time()
    model_shap_time = 0
    shap_values = explainer(background, max_evals=300, batch_size=50, outputs=shap.Explanation.argsort.flip[:1])
    model_shap_time += time.time() - start_time
    for i in range(len(test_generator)):
        img, _ = test_generator[i]
        plt.figure()
        shap.image_plot(shap_values, np.expand_dims(img[0], axis=0), show=False)
        plt.savefig(f"Toy Project/SHAP/shap_{model_name}_{i}.png")
        plt.close()
    shap_times.append(model_shap_time)
    

lime_times = []
explainer = lime_image.LimeImageExplainer()

for model_name, model in zip(['Advanced Model', 'Simple Model', 'InceptionV3 Model'], [advanced_model, simple_model, inception_model]):
    print(f"{model_name} LIME ")
    model_lime_time = 0
    for i in range(len(test_generator)):
        img, _ = test_generator[i]
        start_time = time.time()
        explanation = explainer.explain_instance(
            img[0], 
            model.predict, 
            top_labels=1, 
            hide_color=0, 
            num_samples=1000
        )
        model_lime_time += time.time() - start_time
        temp, mask = explanation.get_image_and_mask(
            label=explanation.top_labels[0], 
            positive_only=True, 
            hide_rest=False
        )
        plt.figure()
        plt.imshow(mark_boundaries(temp, mask))
        plt.title(f"LIME - {model_name}")
        plt.axis('off')
        plt.savefig(f"Toy Project/LIME/lime_{model_name}_{i}.png")  
        plt.close()
    lime_times.append(model_lime_time)

simple_params = simple_model.count_params()
advanced_params = advanced_model.count_params()
inception_params = inception_model.count_params()

models = ['Advanced Model', 'Simple Model', 'InceptionV3 Model']
params = [advanced_params, simple_params, inception_params]
params_million = [p / 1e6 for p in params]

plt.figure(figsize=(8, 6))
plt.plot(params_million, lime_times, marker='o', label='LIME', color='blue', linestyle='-')
plt.plot(params_million, shap_times, marker='o', label='SHAP', color='orange', linestyle='--')
plt.xlabel("Parameter number of Model (M)")
plt.ylabel("Time (Second)")
plt.title("LIME and SHAP")
plt.legend()

for i, lime_time in enumerate(lime_times):
    plt.text(params_million[i], lime_time + 0.1, f"{lime_time:.2f}s", ha='center', fontsize=9)

for i, shap_time in enumerate(shap_times):
    plt.text(params_million[i], shap_time + 0.1, f"{shap_time:.2f}s", ha='center', fontsize=9)

plt.grid(True)
plt.tight_layout()
plt.show()
