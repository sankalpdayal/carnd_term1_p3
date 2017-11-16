import csv
import cv2
import numpy as np

lines =[]
with open('../data/driving_log.csv) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)




