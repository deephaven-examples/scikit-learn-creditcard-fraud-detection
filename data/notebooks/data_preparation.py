# Deephaven imports
from deephaven.pandas import to_table, to_pandas
from deephaven.plot import Figure
from deephaven import read_csv

# Python imports
from sklearn.neighbors import KDTree as kdtree
import pandas as pd
import numpy as np

# Read the CSV file into a Deephaven table
creditcard = read_csv("https://media.githubusercontent.com/media/deephaven/examples/main/CreditCardFraud/csv/creditcard.csv")

def plot_valid_vs_fraud(col_name):
    # Set the creditcard table as global to make sure we can access it for the plot
    global creditcard

    # Make sure the input corresponds to a column
    allowed_col_names = [item for item in range(1, 29)] + ["V" + str(item) for item in range(1, 29)]
    if col_name not in allowed_col_names:
        raise ValueError("The column name you specified is not valid.")
    if isinstance(col_name, int):
        col_name = "V" + str(col_name)

    # Some convenience variables for plotting
    num_valid_bins = 50
    num_fraud_bins = 50
    valid_label = col_name + "_Valid"
    fraud_label = col_name + "_Fraud"
    valid_string = "Class = 0"
    fraud_string = "Class = 1"

    # Create a fancy histogram plot
    valid_vs_fraud = \
        Figure()\
        .plot_xy_hist(series_name=valid_label, t=creditcard.where(valid_string), x=col_name, nbins=num_valid_bins)\
        .x_twin()\
        .plot_xy_hist(series_name=fraud_label, t=creditcard.where(fraud_string), x=col_name, nbins=num_valid_bins)\
        .show()
    return valid_vs_fraud

valid_vs_fraud_V4 = plot_valid_vs_fraud("V4")
valid_vs_fraud_V12 = plot_valid_vs_fraud("V12")
valid_vs_fraud_V14 = plot_valid_vs_fraud("V14")

creditcard = creditcard.select(["Time", "V4", "V12", "V14", "Amount", "Class"])
train_data = creditcard.where(["Time >= 43200 && Time < 57600"])
test_data = to_pandas(creditcard.where(["Time >= 129600 && Time < 144000"]))

# Turn the training data into a Pandas DataFrame
data = to_pandas(train_data.select(["V4", "V12", "V14"])).values

# Get nearest neighbor distances using a K-d tree
tree = kdtree(data)
dists, inds = tree.query(data, k = 2)

# Sort the nearest neighbor distances in ascending order
neighbor_dists = np.sort(dists[:, 1])

x = np.array(range(len(neighbor_dists)))

# Turn our x and y (sorted neighbor distances) into a Deephaven table
nn_dists = pd.DataFrame({"X": x, "Y": neighbor_dists})
nn_dists = to_table(nn_dists)

# Plot the last few hundred points so we can see the "elbow"
neighbor_dists = Figure().plot_xy(series_name="Nearest neighbor distance", t=nn_dists.where("X > 30000"), x="X", y="Y").show()
