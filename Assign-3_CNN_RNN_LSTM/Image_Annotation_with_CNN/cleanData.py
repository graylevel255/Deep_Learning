import csv
import numpy as np
import os

# read images from imageID.csv file and create clean data
features = []
oneHotK = []
with open('Y.csv') as f1:
    csv_reader = csv.reader(f1, delimiter=',')
    for row in csv_reader:
        oneHotK.append(row)

oneHotKClean = []
validImageLoc = []
i = 0
with open('ImageID.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        img_path = 'image/' + row[0].replace('\\', '/')
        if os.path.exists(img_path):
            validImageLoc.append([row[0]])
            oneHotKClean.append(oneHotK[i])
        i += 1

with open("ImageID_clean.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerows(validImageLoc)

with open("Y_clean.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerows(oneHotKClean)
