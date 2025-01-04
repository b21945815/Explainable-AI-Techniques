from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

image_size = (330, 330)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.15 
)

train_generator = train_datagen.flow_from_directory(
    "C:\\Users\\fatih\\OneDrive\\Masaüstü\\562 Machine\\Proje\\Coding\\Toy Project\\Train",
    target_size=image_size,
    batch_size=32,
    class_mode='categorical',
    subset='training'  
)

validation_generator = train_datagen.flow_from_directory(
    "C:\\Users\\fatih\\OneDrive\\Masaüstü\\562 Machine\\Proje\\Coding\\Toy Project\\Train",
    target_size=image_size,
    batch_size=32,
    class_mode='categorical',
    subset='validation'  
)

simple_model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(330, 330, 3)), 
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),  
    MaxPooling2D((2, 2)),

    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(256, (3, 3), activation='relu'), 
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation='relu'),  
    Dropout(0.5),  
    Dense(8, activation='softmax')  
])

checkpoint = ModelCheckpoint(
    filepath='best_simple_model.keras', 
    monitor='val_loss',             
    save_best_only=True,            
    mode='min',                   
    verbose=1
)

simple_model.summary()
simple_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

simple_model_history = simple_model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
    callbacks=[early_stopping, checkpoint]
)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(simple_model_history.history['accuracy'], label='Training Accuracy')
plt.plot(simple_model_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(simple_model_history.history['loss'], label='Training Loss')
plt.plot(simple_model_history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
