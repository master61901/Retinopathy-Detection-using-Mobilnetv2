# Retinopathy-Detection-using-Mobilnetv2
Retinopathy detection using Mobilenetv2 and PyQt5 ans cv2

Pre-requisite :
Install Keras, numpy, sckit-learn, cv2, tensorflow ,Anaconda ,Pandas

GUI Interface :
Spyder 

Working :

Introduction:detection of diabetic retinopathy in body structure image is done by image process and machine learning techniques. Probabilistic Neural Network (PNN) and Support vector machines (SVM) area unit the 2models adopted for detection of diabetic retinopathy in body structure image and their results analyzed and compared.

Dataset:Large set of high-resolution membrane pictures taken underneath a range of imaging conditions. A left and right field is provided for each subject. Images area unit labelled with a topic id moreover as either left or right .A practitioner has rated the presence of diabetic retinopathy in every image on a scale of zero to four, in line with the subsequent scale.
● Zero - No DR
● One - Mild
● Two - Moderate
● Three - Severe
● Four - Severe Most

Preprocessing:
 Image blurred-A noise reduction technique used for blurring the photographs. The pictures are of blurred exploitation a Gaussian operates. 
 Image Resizing-After applying the higher than operations, we'd like to avoid wasting the new pictures within the folder which will be used later. The initial pictures square measure every of around 5 MB, the whole pictures folder occupies 35 GB area. We are able to cutback this by resizing pictures. victimisation multi core threading / Multi process, we are able to attain this task in short span. Resize train images to 512x512.
Image Random flip-This technique helps in achieving more relevant data for the neural network. The images are flipped horizontally, vertically and changed in different aspects such as flip angles, border angles, filter bound to achieve high trained accuracy for CNN.

Learning :
The purpose of using MobileNetv2 was [7]:
1. MobileNetV2 has less parameters, due to which it is easy
to train.
2. Using a pre-trained encoder helps the model to
converge much faster in comparison to the non-pretrained
model
3. A pre-trained encoder helps the model to achieve high
performance as compared to a non pre-trained model

After ripping and process of information, the most necessary step is learning from the coaching information victimisation a MobileNetv2 Convolutional Neural Network (CNN) model, this model helps in correct
mapping of the raw pixels from the pictures collected to the dataset or roentgenography scans .The CNN is employed initial for learning the coaching set pictures till the validation set accuracy is achieved .
Final Result 
![image](https://github.com/user-attachments/assets/dae19bfd-bbf4-4265-8504-91ea483ad06f)



GUI Working 




![image](https://github.com/user-attachments/assets/8db3db70-c6ef-4ba0-8bc4-40f8d584c14d)






