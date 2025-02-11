import os
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras import backend as K

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Defining data form of set
data_train = {'img': [], 'mask': []}
data_test = {'img': [], 'mask': []}

# Data loader function
def LoadData(data_set=None, imgPath=None, maskPath=None, shape=128):
    imgNames = os.listdir(imgPath)  # Get image names
    maskNames = []  # Get mask names   
    for i in imgNames:
        maskNames.append(re.sub('\.png', '_L.png', i))  # L: label
    # Get address 
    imgAddr = imgPath + '/'
    maskAddr = maskPath + '/'
    
    # Divide data
    for i in range(len(imgNames)):
        # Read data
        img = plt.imread(imgAddr + imgNames[i])
        mask = plt.imread(maskAddr + maskNames[i])
        
        # Resize the image as 512x512
        img = cv2.resize(img, (shape, shape)) 
        mask = cv2.resize(mask, (shape, shape))
        
        # Divide data
        data_set['img'].append(img)
        data_set['mask'].append(mask)
        
    return data_set

data_train = LoadData(data_train, imgPath='../UNET/archive/CamVid/train', maskPath='../UNET/archive/CamVid/train_labels', shape=256)
data_test = LoadData(data_test, imgPath='../UNET/archive/CamVid/test', maskPath='../UNET/archive/CamVid/test_labels', shape=256)

# Convert data set to array
data_train_img = np.array(data_train['img'])
data_train_mask = np.array(data_train['mask'])

data_test_img = np.array(data_test['img'])
data_test_mask = np.array(data_test['mask'])

data_train = {'img': data_train_img, 'mask': data_train_mask}
data_test = {'img': data_test_img, 'mask': data_test_mask}

# Double Convolution 
def DoubleConv(input_tensor, nb_filters, ker_size=3):
    # 1st Convolution 
    conv = tf.keras.layers.Conv2D(filters=nb_filters, kernel_size=(ker_size, ker_size), kernel_initializer='he_normal', padding='same')(input_tensor)
    conv = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.Activation('relu')(conv)
    
    # 2nd Convolution
    conv = tf.keras.layers.Conv2D(filters=nb_filters, kernel_size=(ker_size, ker_size), kernel_initializer='he_normal', padding='same')(conv)
    conv = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.Activation('relu')(conv)
    
    return conv

# U-net: there are 2 paths: encoder and decoder (or contracting path and expansive path)
def UNET(input_image, nb_filters=16, drop_out=0.1):
    # Encoder path (or contracting path)
    conv1 = DoubleConv(input_image, nb_filters, ker_size=3)
    max_pool1 = tf.keras.layers.MaxPooling2D((2, 2))(conv1)
    max_pool1 = tf.keras.layers.Dropout(drop_out)(max_pool1)
    
    conv2 = DoubleConv(max_pool1, nb_filters*2, ker_size=3)
    max_pool2 = tf.keras.layers.MaxPooling2D((2, 2))(conv2)
    max_pool2 = tf.keras.layers.Dropout(drop_out)(max_pool2)
    
    conv3 = DoubleConv(max_pool2, nb_filters*4, ker_size=3)
    max_pool3 = tf.keras.layers.MaxPooling2D((2, 2))(conv3)
    max_pool3 = tf.keras.layers.Dropout(drop_out)(max_pool3)
    
    conv4 = DoubleConv(max_pool3, nb_filters*8, ker_size=3)
    max_pool4 = tf.keras.layers.MaxPooling2D((2, 2))(conv4)
    max_pool4 = tf.keras.layers.Dropout(drop_out)(max_pool4)
    
    conv5 = DoubleConv(max_pool4, nb_filters*16, ker_size=3)
    
    # Decoder path (or expansive path)
    upConv1 = tf.keras.layers.Conv2DTranspose(nb_filters*8, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv5)
    upConv1 = tf.keras.layers.concatenate([upConv1, conv4])
    upConv1 = tf.keras.layers.Dropout(drop_out)(upConv1)
    conv6 = DoubleConv(upConv1, nb_filters*8, ker_size=3)
    
    upConv2 = tf.keras.layers.Conv2DTranspose(nb_filters*4, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv6)
    upConv2 = tf.keras.layers.concatenate([upConv2, conv3])
    upConv2 = tf.keras.layers.Dropout(drop_out)(upConv2)
    conv7 = DoubleConv(upConv2, nb_filters*4, ker_size=3)
    
    upConv3 = tf.keras.layers.Conv2DTranspose(nb_filters*2, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv7)
    upConv3 = tf.keras.layers.concatenate([upConv3, conv2])
    upConv3 = tf.keras.layers.Dropout(drop_out)(upConv3)
    conv8 = DoubleConv(upConv3, nb_filters*2, ker_size=3)
    
    upConv4 = tf.keras.layers.Conv2DTranspose(nb_filters, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv8)
    upConv4 = tf.keras.layers.concatenate([upConv4, conv1])
    upConv4 = tf.keras.layers.Dropout(drop_out)(upConv4)
    conv9 = DoubleConv(upConv4, nb_filters*1, ker_size=3)
    
    out_put = tf.keras.layers.Conv2D(3, (1, 1), activation='sigmoid')(conv9)
    model = tf.keras.Model(inputs=[input_image], outputs=[out_put])
    
    return model

# F1 score calculation
def f1_score(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float32'))
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float32'))
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float32'))
    return 2 * tp / (2 * tp + fp + fn)

# Initial Model
input_image = tf.keras.layers.Input((256, 256, 3))
myModel = UNET(input_image, drop_out=0.25)

myModel.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy', 'Precision', 'Recall', f1_score])

# Training Model
trainModel = myModel.fit(data_train['img'], data_train['mask'], epochs=500, verbose=1)

# Save the model
def save_model(model, model_name="my_unet_model.h5"):
    model.save(model_name)
    print(f"Model saved as {model_name}")

# Save training plots
def create_directories(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_training_plots(train_history, save_dir="training_results"):
    create_directories(save_dir)
    
    # Loss curve
    plt.figure(figsize=(12, 10))
    plt.subplot(3, 2, 1)
    plt.plot(train_history.history['loss'], label='Training Loss', color='blue')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "training_loss.png"))
    
    # Accuracy curve
    plt.subplot(3, 2, 2)
    plt.plot(train_history.history['accuracy'], label='Training Accuracy', color='green')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "training_accuracy.png"))
    
    # Precision curve
    plt.subplot(3, 2, 3)
    plt.plot(train_history.history['precision'], label='Training Precision', color='orange')
    plt.title('Training Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "training_precision.png"))
    
    # Recall curve
    plt.subplot(3, 2, 4)
    plt.plot(train_history.history['recall'], label='Training Recall', color='red')
    plt.title('Training Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "training_recall.png"))
    
    # F1 score curve
    if 'f1_score' in train_history.history:
        plt.subplot(3, 2, 5)
        plt.plot(train_history.history['f1_score'], label='Training F1 Score', color='purple')
        plt.title('Training F1 Score')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, "training_f1_score.png"))

    plt.tight_layout()
    plt.show()

# Save model and training results
save_model(myModel, "unet_trained_model.h5")
save_training_plots(trainModel, "training_results")

# Prediction function
def predict(data_test, model, shape=256):
    image = data_test['img']
    mask = data_test['mask']
    
    predictions = model.predict(image)
    for i in range(len(predictions)):
        predictions[i] = cv2.merge((predictions[i,:,:,0], predictions[i,:,:,1], predictions[i,:,:,2]))
    
    return predictions, image, mask

# Display predicted results
def show_predictions(image, prediction, mask):
    plt.figure(figsize=(10,10))
    
    plt.subplot(1,3,1)
    plt.imshow(image)
    plt.title('Image')
    
    plt.subplot(1,3,2)
    plt.imshow(prediction)
    plt.title('Predicted Mask')
    
    plt.subplot(1,3,3)
    plt.imshow(mask)
    plt.title('Actual Mask')
    
    plt.show()

# Get predicted results
predictions, image, mask = predict(data_test, myModel)

# Show some examples
for i in range(5):
    show_predictions(image[i], predictions[i], mask[i])
