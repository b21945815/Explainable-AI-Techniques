import os
import cv2
import imutils
import numpy as np

classes_dir = ["basophil", "eosinophil", "erythroblast", "ig", "lymphocyte", "monocyte", "neutrophil", "platelet"]

file_number = 3500
resize_dims = (330, 330)

root_dir = r"C:/Users/fatih/OneDrive/Masaüstü/562 Machine/Proje/Coding/Toy Project/Database Origin"
target_dir = r"C:/Users/fatih/OneDrive/Masaüstü/562 Machine/Proje/Coding/Toy Project/Database Target"

def augment_image(image, method="rotate"):
    if method == "rotate":
        return imutils.rotate(image, angle=90)
    elif method == "flip":
        return cv2.flip(image, 1) 
    elif method == "crop":
        rows, cols, _ = image.shape
        start_row, start_col = int(rows * 0.1), int(cols * 0.1)  
        end_row, end_col = int(rows * 0.9), int(cols * 0.9)
        cropped_image = image[start_row:end_row, start_col:end_col]
        return cv2.resize(cropped_image, (cols, rows))
    elif method == "blur":
        return cv2.GaussianBlur(image, (5, 5), 0)  
    else:
        return image
    
for cellClass in classes_dir:
    class_path = os.path.join(root_dir, cellClass)
    target_path = os.path.join(target_dir, cellClass)

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    cell_file_list = os.listdir(class_path)
    cell_file_count = len(cell_file_list)
    delta = file_number - cell_file_count

    for k in range(min(cell_file_count, file_number)):
        file = cell_file_list[k]
        file_path = os.path.join(class_path, file)
        image = cv2.imread(file_path)
        resized_image = cv2.resize(image, resize_dims)
        destination_path = os.path.join(target_path, file)
        cv2.imwrite(destination_path, resized_image)

    augment_methods = ["rotate", "flip", "crop", "blur"]
    k = 0
    while delta > 0:
        file = cell_file_list[k % cell_file_count]  
        index = file.rfind('.')
        file_name = file[:index]
        file_extension = file[index:]
        file_path = os.path.join(class_path, file)
        image = cv2.imread(file_path)
        resized_image = cv2.resize(image, resize_dims)

        for method in augment_methods:
            if delta <= 0:
                break
            augmented_image = augment_image(resized_image, method)
            new_file_name = f"{file_name}-{method}-{delta}{file_extension}"
            augmented_destination = os.path.join(target_path, new_file_name)
            cv2.imwrite(augmented_destination, augmented_image)
            delta -= 1

        k += 1  

    print(f"{cellClass} processing completed. Total images: {file_number - delta}")

print("All classes processed successfully.")