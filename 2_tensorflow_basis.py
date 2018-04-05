# input libraries:
import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

# 1 load dataset:
california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")
california_housing_dataframe

# 2 data preprocess:
california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))
california_housing_dataframe["median_house_value"] /= 1000.0
california_housing_dataframe
california_housing_dataframe.describe()

# 3 build the model:
## feature: total_rooms
## predicted value: median_house_value

# 3-1) Define the input feature: total_rooms:
my_feature = california_housing_dataframe[["total_rooms"]]
# Configure a numeric feature column for total_rooms:
feature_columns = [tf.feature_column.numeric_column("total_rooms")]

# 3-2) Define the target:
targets = california_housing_dataframe["median_house_value"]

# 3-3) Using Gradient Descent as the optimizer for traning the model:
my_optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.0000001)
# clip the gradient to ensure the gradient will not become too large!
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
# Configure the linear regression model with out feature columns and optimizer.
# Set a learning rate of 0.0000001 for Gradient Descent.
linear_regressor = tf.estimator.LinearRegressor(
    feature_columns = feature_columns,
    optimizer = my_optimizer
)

# 4 Define input function:
def my_input_fn(features, targets, batch_size = 1, shuffle = True, num_epochs = None):
    # Convert pandas data into a dict of np arrays:
    features = {key:np.array(value) for key,value in dict(features).items()}
    # Construct a dataset and configure batching & repeating
    ds = Dataset.from_tensor_slices((features, targets)) # warning: 2GB limit???
    ds = ds.batch(batch_size).repeat(num_epochs)
    # Shuffle the data
    if shuffle:
        ds = ds.shuffle(buffer_size = 10000)
    # Return the next batch of data:
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

# 5 Train the model:
# tf.estimator.LinearRegressor: steps: the total steps of the iteration(one 
#       iteration causes one revision of the parameters
_ = linear_regressor.train(
    input_fn = lambda:my_input_fn(my_feature, targets),
    steps = 100
)

# 6 Evaluate the model:
# 6-1) Create an input function for prediction:
# Just one prediction for each example(no need to repeat and shuffle data)
prediction_input_fun = lambda: my_input_fn (my_feature,targets, num_epochs = 1, shuffle = False)
# 6-2) Call predict() on linear regrressor to make predictions:
predictions = linear_regressor.predict(input_fn = prediction_input_fun)
# 6-3) Format predictions as NumPy array, so we can calculate error metrics:
predictions = np.array([item['predictions'][0] for item in predictions])

# predictions
# targets
# 6-4) Print Mean Squared Error  and Root Mean Squared Error:  NOT WORKING!!!
mean_squared_error = metrics.mean_squared_error(predictions, targets)
root_mean_squared_error = math.sqrt(mean_squared_error)
print("Mean Squared Error (On training data): %0.3f" % mean_squared_error)
print("Root Squared Error (On training data): %0.3f" % root_mean_squared_error)

# 7) Results display:
calibration_data = pd.DataFrame()
calibration_data["predictions"] = pd.Series(predictions)
calibration_data["targets"] = pd.Series(targets)
calibration_data.describe()
# plt:
sample = california_housing_dataframe.sample(n=300)
# Get the min and max total_rooms values.
x_0 = sample["total_rooms"].min()
x_1 = sample["total_rooms"].max()

# Retrieve the final weight and bias generated during training.
weight = linear_regressor.get_variable_value('linear/linear_model/total_rooms/weights')[0]
bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

# Get the predicted median_house_values for the min and max total_rooms values.
y_0 = weight * x_0 + bias 
y_1 = weight * x_1 + bias
# Plot our regression line from (x_0, y_0) to (x_1, y_1).
plt.plot([x_0, x_1], [y_0, y_1], c='r')

# Label the graph axes.
plt.ylabel("median_house_value")
plt.xlabel("total_rooms")

# Plot a scatter plot from our data sample.
plt.scatter(sample["total_rooms"], sample["median_house_value"])

# Display graph.
plt.show()