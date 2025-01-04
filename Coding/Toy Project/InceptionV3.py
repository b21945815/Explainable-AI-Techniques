from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, AveragePooling2D, BatchNormalization, Activation
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import RMSprop
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

pre_trained_model = InceptionV3(input_shape=(330, 330, 3), include_top=False, weights="imagenet")

for layer in pre_trained_model.layers: # [:209]
    layer.trainable = False

x = pre_trained_model.output
x = AveragePooling2D(pool_size=(2, 2), strides=2, padding="valid")(x)
x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(256)(x)  
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)
output = Dense(8, activation="softmax")(x)

model = Model(inputs=pre_trained_model.input, outputs=output)

checkpoint = ModelCheckpoint(
    filepath='best_advanced_model.keras', 
    monitor='val_loss',             
    save_best_only=True,            
    mode='min',                   
    verbose=1
)
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

model.summary()

initial_learning_rate = 1e-4
lr_schedule = ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
optimizer = RMSprop(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

simple_model_history = model.fit(
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
