import numpy as np
import keras
from keras import backend as K
from keras.layers.core import Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing.image import img_to_array, array_to_img, load_img
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import cv2
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
from itertools import chain 


K.tensorflow_backend._get_available_gpus()
    
    # 0: 1.0,   # normal
    # 1: 10.0,  # Mild_NPDR
    # 2: 4.0,   # Moderate_NPDR
    # 3: 29.0,  # Severe_NPDR
    # 4: 36.0,  # PDR 
    
DISEASE_CLASSES ={
  0: 'Normal',
  1: 'Mild NPDR',
  2: 'Moderate NPDR',
  3: 'Severe NPDR',
  4: 'Proliferative DR'  
}

def Load_Training_Model():
    # Create a MobileNet model
    mobile = keras.applications.mobilenet.MobileNet()   
    x = mobile.layers[-6].output
    x = Dropout(0.25)(x)
    predictions = Dense(5, activation='softmax')(x)
    model = Model(inputs=mobile.input, outputs=predictions)

    for layer in model.layers[:-23]:
        layer.trainable = False

    model.load_weights('model.h5')
    
    return model

def Predict_Test_Image_File(model):

    root = tk.Tk()
    root.withdraw()
    imageFileName = filedialog.askopenfilename()

    image = cv2.imread(imageFileName)
        
    # pre-process the image for classification
    image = cv2.resize(image, (224, 224))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image)

    print(predictions)

    final_prediction = predictions.argmax(axis=1)[0]

    print(final_prediction)

    disease_name = DISEASE_CLASSES[final_prediction]

    print(disease_name)        
    
    tk.messagebox.showinfo('Test Image Prediction',disease_name)    
    
    Show_Bar_Chart(predictions)
    
def Show_Bar_Chart(predictions_list):
    
    objects = ('normal', 'Mild_NPDR','Moderate_NPDR', 'Severe_NPDR', 'PDR')
    y_pos = np.arange(len(objects))
    
    flatten_predictions_list = list(chain.from_iterable(predictions_list)) 
    flatten_predictions_percentages = map(lambda x: round(x * 100, 2), flatten_predictions_list)    
        
    performance = list(flatten_predictions_percentages)
    print(performance)
    
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    
    for index,data in enumerate(performance):
        plt.text(x=index , y =data+1 , s=f"{data}" , fontdict=dict(fontsize=10))
        
    plt.xticks(y_pos, objects)
    plt.ylabel('Percentage Probabilities')
    plt.title('Disease Predictions')

    plt.show()
    


if __name__ == "__main__":
        
    model_definition = Load_Training_Model()
    
    root= tk.Tk() # create window
    root.withdraw()

    MsgBox = tk.messagebox.askquestion ('Tensorflow Predictions','Do you want to test Images for Predictions')
    
    while MsgBox == 'yes':
        MsgBox = tk.messagebox.askquestion ('Test Image','Do you want to test new Image')
        if MsgBox == 'yes':
            Predict_Test_Image_File(model_definition)            
        else:
            tk.messagebox.showinfo('EXIT', "Exiting the Application")
            break
        


