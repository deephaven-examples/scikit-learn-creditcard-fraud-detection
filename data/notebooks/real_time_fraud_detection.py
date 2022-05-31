start_time = dbtu.to_datetime("2021-11-17T12:00:00 NY")
end_time = dbtu.to_datetime("2021-11-17T16:01:00 NY")

# Replay the test_data table
test_data_replayer = TableReplayer(start_time, end_time)
creditcard_live = test_data_replayer.add_table(test_data, "TimeStamp")
test_data_replayer.start()
creditcard_live = creditcard_live.view(["Time", "V4", "V12", "V14", "Amount", "Class"])

# A function to place new observations into our existing clusters
def dbscan_predict(X_new):
    n_rows = X_new.shape[0]
    data_with_new = np.vstack([data, X_new])
    tree = kdtree(data_with_new)
    dists, points = tree.query(data_with_new, 10)
    dists = dists[-n_rows:]
    detected_fraud = [0] * n_rows
    for idx in range(len(dists)):
        if any(dists[idx] > 1):
            detected_fraud[idx] = 1
    return np.array(detected_fraud)

# A function to gather data from a Deephaven table into a NumPy array
def table_to_numpy(rows, cols):
    return gather.table_to_numpy_2d(rows, cols, np_type = np.double)

# A function to scatter data back into an output table
def scatter(data, idx):
    return data[idx]

predicted_fraud_live = learn.learn(
    table = creditcard_live,
    model_func = dbscan_predict,
    inputs = [learn.Input(["V4", "V12", "V14"], table_to_numpy)],
    outputs = [learn.Output("PredictedClass", scatter, "short")],
    batch_size = 1000
)

dbscan_correct_valid = predicted_fraud_live.where(["PredictedClass == 0 && Class == 0"])
dbscan_correct_fraud = predicted_fraud_live.where(["PredictedClass == 1 && Class == 1"])
dbscan_false_positive = predicted_fraud_live.where(["PredictedClass == 1 && Class == 0"])
dbscan_false_negative = predicted_fraud_live.where(["PredictedClass == 0 && Class == 1"])
