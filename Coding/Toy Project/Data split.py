import os
import numpy as np
import shutil

classes_dir = ["basophil", "eosinophil", "erythroblast", "ig", "lymphocyte", "monocyte", "neutrophil", "platelet"]

target_dir = r"C:/Users/fatih/OneDrive/Masaüstü/562 Machine/Proje/Coding/Toy Project/Database Target/"
train = r"C:/Users/fatih/OneDrive/Masaüstü/562 Machine/Proje/Coding/Toy Project/Train/"
test = r"C:/Users/fatih/OneDrive/Masaüstü/562 Machine/Proje/Coding/Toy Project/Test/"
test_ratio = 0.25
# https://stackoverflow.com/questions/57394135/split-image-dataset-into-train-test-datasets
def split(class_name):
    train_class_path = train + class_name
    test_class_path = test + class_name
    if not os.path.exists(train_class_path):
        os.makedirs(train_class_path)
    if not os.path.exists(test_class_path):
        os.makedirs(test_class_path)
    source = target_dir + class_name
    all_file_names = os.listdir(source)
    np.random.shuffle(all_file_names)
    train_file_names, test_file_names = np.split(np.array(all_file_names), [int(len(all_file_names) * (1 - test_ratio))])
    train_file_names = [source + '/' + name for name in train_file_names.tolist()]
    test_file_names = [source + '/' + name for name in test_file_names.tolist()]
    print("*****************************")
    print('Total images: ', len(all_file_names))
    print('Training: ', len(train_file_names))
    print('Testing: ', len(test_file_names))
    print("*****************************")
    for name in train_file_names:
        shutil.copy(name, train_class_path)

    for name in test_file_names:
        shutil.copy(name, test_class_path)


for cell_class in classes_dir:
    split(cell_class)
    print(f"Copying Done for class {cell_class}!")

