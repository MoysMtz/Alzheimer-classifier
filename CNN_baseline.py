import numpy as np
import random
import matplotlib.pyplot as plt
import os
import cv2 as cv      
import seaborn as sns
import pandas as pd
import time
from utils import get_model, get_metrics, get_data_set, plotConfusionMatrix
import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
from keras.utils import to_categorical

np.random.seed(100)
image_directory="C:/Users/MM/OneDrive/Documentos/Alzheimer/Data_audio/Spectograms_hsv/"
train_data, train_label = get_data_set(image_directory)
train_label = to_categorical(train_label, num_classes=3)
x_train, x_validation, x_train_label, x_validation_label = train_test_split(train_data, train_label, test_size=0.15, shuffle=True, random_state=0)
del train_data, train_label
kfold = KFold(n_splits=5, shuffle=True, random_state=0)

Scoring = {}
Accuracy, AUC, F1, Precision, Recall,  Time, Loss = [], [], [], [], [], [], []
Histories = {}
histories_loss, histories_val_loss, histories_acc, histories_val_acc = [],[],[],[]

fold_no = 1
class_names = ['AD','MCI','HC']
for train,test in kfold.split(x_train,x_train_label):
    
    print(f'Training for fold {fold_no} ...')
    
    model = get_model()
    
    optimazer = tf.keras.optimizers.SGD(learning_rate=0.0001)
    model.compile(optimizer = optimazer,                    
        loss= 'categorical_crossentropy',   
        metrics= ['accuracy']) 
   
    callbacks = [ 
        keras.callbacks.EarlyStopping(    
        monitor = 'val_accuracy',   
        patience = 15),     
  
        keras.callbacks.ModelCheckpoint(
        filepath=f"alzheimer{fold_no}.keras",   
        save_best_only=True,
        monitor="val_accuracy")  
    ]
    
    start = time.time()
    
    history = model.fit(x_train[train], x_train_label[train], batch_size = 128, epochs = 100, verbose=0, 
                        validation_data = (x_train[test],x_train_label[test]), callbacks=callbacks)
    
    end = time.time()
    
    Time.append(end-start)
    histories_loss.append(history.history['loss'])
    histories_val_loss.append(history.history['val_loss'])
    histories_acc.append(history.history['accuracy'])
    histories_val_acc.append(history.history['val_accuracy'])

    test_model = keras.models.load_model(f"alzheimer{fold_no}.keras")
    loss, test_acc = test_model.evaluate(x_validation,x_validation_label, verbose = 0)
    
    Loss.append(loss)
    Accuracy.append(test_acc)
    
    #--------------------------------------------------------------------------------------------------
    prediction = test_model.predict(x_validation)

    #----------------------------------------------------------------------------------------------
     
    area_roc,f1,precision,recall = get_metrics(class_names,x_validation_label,prediction)
    
    AUC.append(area_roc), F1.append(f1), Precision.append(precision), Recall.append(recall)
    
    fold_no += 1
    #---------------------------------------------------------------------------------------------------

        
Scoring['Accuracy'], Scoring['AUC'], Scoring['F1'], Scoring['Precision'], Scoring['Recall'], Scoring['Time'], Scoring['Loss'] = Accuracy, AUC, F1, Precision, Recall, Time, Loss
Scoring = pd.DataFrame(Scoring)

Histories['loss'], Histories['val_loss'], Histories['accuracy'], Histories['val_accuracy'] = histories_loss, histories_val_loss, histories_acc, histories_val_acc 
Histories = pd.DataFrame(Histories)

Scoring.to_csv('C:/Users/MM/Documents/Alzhemimer/Scores.csv', index = False, encoding = 'utf-8')
Histories.to_csv('C:/Users/MM/Documents/Alzhemimer/Histories.csv', index = False, encoding = 'utf-8')

test_model = keras.models.load_model("C:/Users/MM/Documents/Alzheimer/CNN/alzheimer.keras")
prediction = test_model.predict(x_validation)
y_prediction = []
y_truth = []
for i in range(len(prediction)):
    y_prediction.append(np.argmax(prediction[i]))
    y_truth.append(np.argmax(x_validation_label[i]))
y_prediction = np.array(y_prediction)
y_truth = np.array(y_truth)

plotConfusionMatrix(y_truth, y_prediction)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(class_names)):
    fpr[i], tpr[i], _ = roc_curve(x_validation_label[:,i], prediction[:,i])
    roc_auc[i] = auc(fpr[i], tpr[i])
plt.figure()
plt.style.use('default')
colors = ["#FF3B9D",'#A02B93', '#0F9ED5']

for i in range(len(class_names)):
    plt.plot(fpr[i], tpr[i], lw=2, color = colors[i % len(colors)],
             label=' {0} (AUC = {1:0.2f})'''.format(class_names[i], roc_auc[i]*100))

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.grid(False)
plt.legend(prop = {'size': 12}, loc='lower right')
plt.show()

output_image_path = os.path.join('C:/Users/MM/Documents/Alzheimer', "ROC_Curve.png")
plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0, dpi=300) 
