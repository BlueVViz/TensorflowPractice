import tensorflow as tf
import pandas as pd


inputData = pd.read_csv("./data/bear_data.csv", usecols=[0, 1])
outputData = pd.read_csv("./data/bear_data.csv", usecols=[2, 3])

print(inputData)
print(outputData)