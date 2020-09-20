from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np
import glob
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import imageio as im
import csv
import os.path


base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
features = []

with open('ImageID_clean_small.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        #  img_path = 'image/' + row[0].replace('\\', '/')
        img_path = 'image\\' + row[0]
        print("Loading Image " + img_path)
        img = image.load_img(img_path, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        features.append(model.predict(img_data).reshape(4096))

features = np.array(features)
np.savetxt("features_VGG16.csv", features, delimiter=",")

# Visualizing Convolution Layer
# Extracts the outputs of the top 12 layers
#layer_outputs = [layer.output for layer in base_model.layers[:1]]
# Creates a model that will return these outputs, given the model input
# activation_model = Model(inputs=base_model.input, outputs=base_model.layers[1].output)
# activations = activation_model.predict(img_data)
# first_layer_activation = activations[0]
# print(first_layer_activation.shape)
# plt.matshow(first_layer_activation[ :, :, 4], cmap='viridis')
#
# images_per_row = 16
#
# for layer_activation in activations:
#     n_features = layer_activation.shape[-1] # number of feature maps
#     size = layer_activation.shape[1]
#     n_cols = n_features
#     display_grid = np.zeros((size * n_cols, images_per_row * size))
#     for col in range(n_cols):  # Tiles each filter into a big horizontal grid
#         for row in range(images_per_row):
#             channel_image = layer_activation[:, :, col-1 * images_per_row + row]
#             channel_image = np.clip(channel_image, 0, 255).astype('uint8')
#             display_grid[col * size: (col + 1) * size,  # Displays the grid
#             row * size: (row + 1) * size] = channel_image
#         scale = 1. / size
#         plt.figure(figsize=(scale * display_grid.shape[1],
#                             scale * display_grid.shape[0]))
#         plt.title('Conv1')
#         plt.grid(False)
#         plt.imshow(display_grid, aspect='auto', cmap='viridis')