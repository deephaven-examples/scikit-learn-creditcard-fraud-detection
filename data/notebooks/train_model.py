# Deephaven imports
from deephaven.TableManipulation import Replayer
import deephaven.DateTimeUtils as dbtu
from deephaven import read_csv
from deephaven import dataFrameToTable
from deephaven import tableToDataFrame
from deephaven.learn import gather
from deephaven import learn

# Python imports
from sklearn.neighbors import KDTree as kdtree
from sklearn.cluster import DBSCAN as dbscan
import pandas as pd
import numpy as np
import scipy as sp

# Read external data, remove unwanted parts, and split into train/test
creditcard = read_csv("https://media.githubusercontent.com/media/deephaven/examples/main/CreditCardFraud/csv/creditcard.csv")
creditcard = creditcard.select("Time", "V4", "V12", "V14", "Amount", "Class")
train_data = creditcard.where("Time >= 43200 && Time < 57600")
test_data = creditcard.where("Time >= 129600 && Time < 144000")

# This base time will be used to generate time stamps
base_time = dbtu.convertDateTime("2021-11-16T00:00:00 NY")

# This function will create a timestamp column from the time offset column
def timestamp_from_offset(t):
    global base_time
    db_period = "T{}S".format(t)
    return dbtu.plus(base_time, dbtu.Period(db_period))

# Add a timestamp column to the test data for later replay
test_data = test_data.update("TimeStamp = (DateTime)timestamp_from_offset(Time)")

# This placeholder will be replaced by our trained DBSCAN model
db = 0

# A function to apply DBSCAN with eps = 1 and min_samples = 10
def perform_dbscan(data):
    global db
    db = dbscan(eps = 1, min_samples = 10).fit(data)
    return db.labels_

# Our gather function for DBSCAN
def dbscan_gather(rows, cols):
    return gather.table_to_numpy_2d(rows, cols, dtype = np.double)

# Our scatter function for DBSCAN
def dbscan_scatter(data, idx):
    if data[idx] == -1:
        data[idx] = 1
    return data[idx]

# Perform DBSCAN on our train_data table
clustered = learn.learn(
    table = train_data,
    model_func = perform_dbscan,
    inputs = [learn.Input(["V4", "V12", "V14"], dbscan_gather)],
    outputs = [learn.Output("PredictedClass", dbscan_scatter, "short")],
    batch_size = train_data.size()
)

# Split DBSCAN guesses (correct and incorrect) into separate tables
dbscan_correct_valid = clustered.where("Class == 0 && PredictedClass == 0")
dbscan_correct_fraud = clustered.where("Class == 1 && PredictedClass == 1")
dbscan_false_positives = clustered.where("Class == 0 && PredictedClass == 1")
dbscan_false_negatives = clustered.where("Class == 1 && PredictedClass == 0")

# Report the accuracy of the model
print("DBSCAN guesses valid - correct! " + str(dbscan_correct_valid.size()))
print("DBSCAN guesses fraud - correct! " + str(dbscan_correct_fraud.size()))
print("DBSCAN guesses valid - wrong! " + str(dbscan_false_positives.size()))
print("DBSCAN guesses fraud - wrong! " + str(dbscan_false_negatives.size()))
