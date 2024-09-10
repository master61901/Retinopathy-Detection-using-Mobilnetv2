# Import the libraries
import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import os
from sklearn.model_selection import train_test_split
import shutil


# Create a new directory for the images
base_dir = 'base_dir'
os.mkdir(base_dir)

# Training file directory
train_dir = os.path.join(base_dir, 'train_dir')
os.mkdir(train_dir)

# Validation file directory
val_dir = os.path.join(base_dir, 'val_dir')
os.mkdir(val_dir)

# Create new folders in the training directory for each of the classes
Normal = os.path.join(train_dir, '0')
os.mkdir(Normal)
Mild_NPDR = os.path.join(train_dir, '1')
os.mkdir(Mild_NPDR)
Moderate_NPDR = os.path.join(train_dir, '2')
os.mkdir(Moderate_NPDR)
Severe_NPDR = os.path.join(train_dir, '3')
os.mkdir(Severe_NPDR)
PDR = os.path.join(train_dir, '4')
os.mkdir(PDR)


# Create new folders in the validation directory for each of the classes

Normal = os.path.join(val_dir, '0')
os.mkdir(Normal)
Mild_NPDR = os.path.join(val_dir, '1')
os.mkdir(Mild_NPDR)
Moderate_NPDR = os.path.join(val_dir, '2')
os.mkdir(Moderate_NPDR)
Severe_NPDR = os.path.join(val_dir, '3')
os.mkdir(Severe_NPDR)
PDR = os.path.join(val_dir, '4')
os.mkdir(PDR)



# Read the metadata
df = pd.read_csv('./dataset/trainLabels.csv')

# Display some information in the dataset
print(df.head())

# Set y as the labels
y = df['level']

# Split the metadata into training and validation
df_train, df_val = train_test_split(df, test_size=0.1, random_state=101, stratify=y)

# Print the shape of the training and validation split
print(df_train.shape)
print(df_val.shape)

# Find the number of values in the training and validation set
df_train['level'].value_counts()
df_val['level'].value_counts()

# Transfer the images into folders
# Set the image id as the index
df.set_index('image', inplace=True)

# Get a list of images in each of the two folders
folder_1 = os.listdir('./dataset/Training')

# Get a list of train and val images
train_list = list(df_train['image'])
val_list = list(df_val['image'])

# Transfer the training images
for image in train_list:

    fname = image + '.jpeg'
    label = df.loc[image, 'level']

    if fname in folder_1:
        # source path to image
        src = os.path.join('./dataset/Training/', fname)
        # destination path to image
        dst = os.path.join(train_dir, str(label), fname)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)
    

# Transfer the validation images
for image in val_list:

    fname = image + '.jpeg'
    label = df.loc[image, 'level']

    if fname in folder_1:
        # source path to image
        src = os.path.join('./dataset/Training/', fname)
        # destination path to image
        dst = os.path.join(val_dir, str(label), fname)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)
    

# Check how many training images are in each folder
print(len(os.listdir('base_dir/train_dir/0')))
print(len(os.listdir('base_dir/train_dir/1')))
print(len(os.listdir('base_dir/train_dir/2')))
print(len(os.listdir('base_dir/train_dir/3')))
print(len(os.listdir('base_dir/train_dir/4')))


# Check how many validation images are in each folder
print(len(os.listdir('base_dir/val_dir/0')))
print(len(os.listdir('base_dir/val_dir/1')))
print(len(os.listdir('base_dir/val_dir/2')))
print(len(os.listdir('base_dir/val_dir/3')))
print(len(os.listdir('base_dir/val_dir/4')))


# Augment the data
class_list = ['0', '1', '2', '3', '4']

for item in class_list:

    # Create a temporary directory for the augmented images
    aug_dir = 'aug_dir'
    os.mkdir(aug_dir)

    # Create a directory within the base dir to store images of the same class
    img_dir = os.path.join(aug_dir, 'img_dir')
    os.mkdir(img_dir)

    # Choose a class
    img_class = item

    # List all the images in the directory
    img_list = os.listdir('base_dir/train_dir/' + img_class)

    # Copy images from the class train dir to the img_dir
    for fname in img_list:
        # source path to image
        src = os.path.join('base_dir/train_dir/' + img_class, fname)
        # destination path to image
        dst = os.path.join(img_dir, fname)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)

    # point to a dir containing the images and not to the images themselves
    path = aug_dir
    save_path = 'base_dir/train_dir/' + img_class

    # Create a data generator to augment the images in real time
    datagen = ImageDataGenerator(
        rotation_range=180,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        # brightness_range=(0.9,1.1),
        fill_mode='nearest')

    batch_size = 50

    aug_datagen = datagen.flow_from_directory(path,
                                              save_to_dir=save_path,
                                              save_format='jpeg',
                                              target_size=(224, 224),
                                              batch_size=batch_size)

    # Generate the augmented images and add them to the training folders
    num_aug_images_wanted = 1000  # total number of images we want to have in each class
    num_files = len(os.listdir(img_dir))
    num_batches = int(np.ceil((num_aug_images_wanted - num_files) / batch_size))

    # run the generator and create about 6000 augmented images
    for i in range(0, num_batches):
        imgs, labels = next(aug_datagen)

    # delete temporary directory with the raw image files
    shutil.rmtree('aug_dir')

# Check how many training images are in each folder
print(len(os.listdir('base_dir/train_dir/0')))
print(len(os.listdir('base_dir/train_dir/1')))
print(len(os.listdir('base_dir/train_dir/2')))
print(len(os.listdir('base_dir/train_dir/3')))
print(len(os.listdir('base_dir/train_dir/4')))


# Check how many validation images are in each folder
print(len(os.listdir('base_dir/val_dir/0')))
print(len(os.listdir('base_dir/val_dir/1')))
print(len(os.listdir('base_dir/val_dir/2')))
print(len(os.listdir('base_dir/val_dir/3')))
print(len(os.listdir('base_dir/val_dir/4')))
