from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
import numpy as np
import glob
import os
from scipy import stats

features = []
i = 0
data_directory = '/home/ubantu/Desktop/DL_A3 Data/classification task/my_class_data/'
directories = [d for d in os.listdir(data_directory)
            if os.path.isdir(os.path.join(data_directory, d))]
print(directories)

for d in directories:
    for image_file in glob.iglob(data_directory + d + '/*.jpg'):
        print(i)
        i += 1


base_model = InceptionV3(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)

# for i in range(images)

# chotadata = '/home/ubantu/Desktop/DL_A3 Data/chota_data/'

# for d in directories:

# directories = [d for d in os.listdir(chotadata)
#              if os.path.isdir(os.path.join(chotadata, d))]
# i = 0
# for image_file in glob.iglob(d + '/*.jpg'):
#     print(i)
#     i += 1

for d in directories:
    for image_file in glob.iglob(data_directory + d + '/*.jpg'):
        img = image.load_img(image_file, target_size=(299, 299))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        feats = model.predict(img_data)
        feats = feats.reshape(feats.shape[1])
        features.append(feats)

incepv3_features = np.array(features)
print(incepv3_features.shape)
print(type(incepv3_features))
print(incepv3_features)

norm_feats = stats.zscore(incepv3_features  , axis=1, ddof=1)

# Save features in a csv file

dirname = os.path.dirname(__file__)
with open(dirname+'/InceptionV3_features.txt', 'w') as f:
    for item in np.array(incepv3_features):
        for value in item:
            f.write("%s " % value)
        f.write("\n")
print("Features Extracted Successfully\n")

with open(dirname+'/NormInceptionV3_features.txt', 'w') as f:
    for item in np.array(norm_feats):
        for value in item:
            f.write("%s " % value)
        f.write("\n")
print("Normalized Features Extracted Successfully\n")