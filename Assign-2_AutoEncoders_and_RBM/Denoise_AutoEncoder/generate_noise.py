import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # to give away the GPU warning in TF
from pandas import read_csv
from matplotlib import pyplot as plt
import numpy as np
import copy

my_path = os.path.dirname(os.path.realpath(__file__))  #gets current path of the file
filename = my_path + "/data.csv"   #gets path of the file
df = read_csv(filename, header = None)
#print(df)
print("DATAFRAME :")
print(df.shape)

images = df.loc[:, :783]
labels = df.loc[:, 784:]

clean_images = images.values
print("\nCLEAN IMAGES :")

print(type(clean_images))
print(clean_images.shape)

print("\n", type(images), type(labels))
print(images.shape)

# Visualize original images

for i in range(2):
    sample = clean_images[i, :]
    print(sample)
    sample = sample.reshape(28, 28)
    plt.gray()
    plt.imshow(sample)
    plt.show()
    print("Original image : ")
    print(sum(sum(sample)))
    print(784 - sum(sum(sample)))


# Add noise and visualize noisy images

N = clean_images.shape[1]
print("Dimension : ", N)
n = int(0.05 * N)  # no. of noisy pixels
print("Number of noisy pixels : ", n)
pixels = N
noisy_images = copy.deepcopy(clean_images)



# Generate noisy pixels
seed = 10
np.random.seed(seed)  # first call
random_pixels = np.random.randint(0, 784, size = n)
print("RANDOM PIXEL VALUES ", random_pixels)
seed += 1
s = 28  # set random value of seed so that it flips the same bits each time for the same image
examples = clean_images.shape[0]
print(examples)
for k in range(examples):
    for i in range(n):
        np.random.seed(s)  # first call
        noise = np.random.randint(784)
        print("SEED  VALUES : ")
        print(noise)
        s += 1
        noisy_images[k, noise] = (1 - noisy_images[k, noise])
    # print("Noisy image : ")
    # print(sum(noisy_images[k, :]))
    # print(784 - sum(noisy_images[k, :]))
    # reshaped_noisy_image = noisy_images[k, :].reshape(28, 28)
    # print(reshaped_noisy_image.shape)
    #im_noisy_reshaped = im_noisy[k, :].reshape(28, 28)
    # plt.gray()
    # plt.imshow(reshaped_noisy_image)
    # plt.show()

print(clean_images[1,:])
print(noisy_images[1,:])
print(clean_images[1,:] - noisy_images[1, :])



np.savetxt(my_path + "/noisy_images_5.csv", noisy_images, fmt='%i', delimiter=",")
# np.savetxt(my_path + "/labels_10.csv", labels, fmt='%i')
# np.savetxt(my_path + "/clean_images.csv", clean_images, fmt='%i', delimiter=",")