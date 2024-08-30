import numpy as np
import random
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import os
import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
import cv2 as cv     
import pandas as pd
from sklearn.metrics import roc_curve, auc, f1_score, recall_score,precision_score, accuracy_score, confusion_matrix
from tensorflow.keras.layers import Reshape, Attention

def get_model():
    inputs = keras.Input(shape=(256,256,3))
    x = keras.layers.Conv2D(16,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu')(inputs)
    x = keras.layers.MaxPooling2D((2,2))(x) 
    x = keras.layers.BatchNormalization(axis=-1)(x)
    x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Conv2D(32,kernel_size=(3,3), strides=(1, 1), padding="same", activation='relu')(x)
    x = keras.layers.MaxPooling2D((2,2))(x)
    x = keras.layers.BatchNormalization(axis=-1)(x)
    x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Conv2D(64,kernel_size=(3,3), strides=(1, 1), padding="same", activation='relu')(x)
    x = keras.layers.MaxPooling2D((2,2))(x)
    x = keras.layers.BatchNormalization(axis=-1)(x)
    x = keras.layers.Dropout(0.3)(x)
    
    #td = Reshape([32,32*64])(x) #2^(#capas cnv) #[256/2^{num_capas_conv},256/2^{num_capas_conv}*num_canales_ultima_capa]

    #x = keras.layers.LSTM(64, return_sequences=False,activation='tanh')(td)
    #x = keras.layers.BatchNormalization()(x)
    #x = keras.layers.Dropout(0.2)(x)
    
    #x = Attention()([x, x])
    
    x = keras.layers.Flatten()(x)

    x = keras.layers.Dense(64,activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Dense(32, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(rate=0.2)(x)

    x = keras.layers.Dense(3, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=x)
    
    return model

def get_data_set(image_directory):
    
    images=os.listdir(image_directory)

    x_train=[]
    x_label=[]

    for image_name in images:
        image_path =image_directory + image_name
        image = cv.imread(image_path)            
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        x_train.append(np.array(image))                          

        image_filename = os.path.splitext(os.path.basename(image_name))[0]

        posicion_AD=image_filename.find('AD')
        posicion_MCI=image_filename.find('MCI')
        posicion_HC=image_filename.find('HC')
        if posicion_AD==0:
            x_label.append(0)     # 0 = AD

        elif posicion_MCI==0:
            x_label.append(1)     # 1 = MCI

        elif posicion_HC==0:
            x_label.append(2)     # 2 = HC
            
    x_label = np.array(x_label)    
    x_train = np.array(x_train, dtype="float32")/255.0
    return x_train, x_label

def get_metrics(class_names,x_validation_label,prediction):
    fpr = dict()
    tpr = dict()
    f1 = dict()
    recall = dict()
    precision = dict()
    area_roc = dict()

    for i in range(len(class_names)):
        
        fpr[i], tpr[i],_ = roc_curve(x_validation_label[:, i], prediction[:, i])
        area_roc[class_names[i]] = auc(fpr[i], tpr[i])
        
        f1[class_names[i]] = f1_score(x_validation_label[:,i], np.round(prediction[:, i]),average='macro')
        
        recall[class_names[i]] = recall_score(x_validation_label[:,i], np.round(prediction[:, i]), average='macro')
        
        precision[class_names[i]] = precision_score(x_validation_label[:,i], np.round(prediction[:, i]), average='macro')
    
    return area_roc,f1,precision,recall


def plotConfusionMatrix(y_true,y_pred):
    conf_matrix = confusion_matrix(y_true,y_pred)
    norm_array = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:,np.newaxis]
    group_counts = ["{0:0.0f}".format(value) for value in conf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in norm_array.flatten()]
    labels = [f"{v1}\n\n{v2}" for v1, v2 in zip(group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(3,3)
    df_cm = pd.DataFrame(conf_matrix, range(3), range(3))
    ax = sn.heatmap(df_cm, annot=labels,fmt='', cmap='Blues')
    ax.set_title('CNN-LSTM-Attention model confusion matrix');
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label');
    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['AD','MCI','HC'])
    ax.yaxis.set_ticklabels(['AD', 'MCI','HC'],va="center")
    #plt.show()
    
    output_image_path = os.path.join('C:/Users/MM/Documents/Alzheimer', "matrix_CNN_LSTM_Attention_conf.png")
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0, dpi=300) 
