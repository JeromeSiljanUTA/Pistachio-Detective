import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class CustomCallback(keras.callbacks.Callback):     # inherits from the keras class
    def on_epoch_end(self, epoch, logs=None):
        print('')
        print(logs.keys())                          # has information such as accuracy, loss, etc.
        print('')
        if logs.get('val_accuracy') > 0.98:
            self.model.stop_training = True

model = keras.Sequential([
    layers.Input((img_height, img_width, 3)),       # using color so last value is 3 (rgb)
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.Flatten(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.Flatten(),
    layers.Dense(50, activation='softmax')
    layers.Dropouts(0.25)
    layers.Dense(10, activation='softmax')
])

train_ds = ImageDataGenerator(
    validation_split=0.1, 
    rescale=1/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
    subset='training',
)

val_ds = ImageDataGenerator(
    validation_split=0.1, 
    rescale=1/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
    subset='validation',
)

train_generator = train_datagen.flow_from_directory(
    'images/',  
    target_size=(600, 600),  
    batch_size=50,
    class_mode='binary',
    subset='training')

model.compile(
    optimizer = tf.optimizers.Adam(), 
    loss = 'binary_crossentropy', 
    metrics=['accuracy'],
)

model.fit(
    train_generator,
    epochs = 15,
    validation_data = val_ds,
    callbacks=[CustomCallback()],
)
