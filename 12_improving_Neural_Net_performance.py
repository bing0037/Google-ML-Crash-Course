# Ref: 7_SelectYourFeature&training&validatoin.py
# Improving Neural Net Performance:
#   1) Normalize input features on the same linear scale;
#   2) Try another Optimizer.
# To test online: https://colab.research.google.com/notebooks/mlcc/improving_neural_net_performance.ipynb?hl=en#scrollTo=XATNnzi3MXP0
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

# Data:
california_housing_dataframe = pd.read_csv("/home/roboticslab16/code/study/Google ML Crash Course/california_housing_train.csv", sep=",")
# california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")
# Reindex the data:
california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))


# 1 Data preprocess: features & targets
def preprocess_features(california_housing_dataframe):
  selected_features = california_housing_dataframe[
    ["latitude",
     "longitude",
     "housing_median_age",
     "total_rooms",
     "total_bedrooms",
     "population",
     "households",
     "median_income"]]
  processed_features = selected_features.copy()
  # Create a synthetic feature.
  processed_features["rooms_per_person"] = (
    california_housing_dataframe["total_rooms"] /
    california_housing_dataframe["population"])
  return processed_features

def preprocess_targets(california_housing_dataframe):
  output_targets = pd.DataFrame()
  # Scale the target to be in units of thousands of dollars.
  output_targets["median_house_value"] = (
    california_housing_dataframe["median_house_value"] / 1000.0)
  return output_targets

# 2 Extract useful data:
training_examples = preprocess_features(california_housing_dataframe.head(12000))
training_examples.describe()
training_targets = preprocess_targets(california_housing_dataframe.head(12000))
training_targets.describe()
validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
validation_examples.describe()
validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))
validation_targets.describe()
# 2-2) Display the data extraction:
plt.figure(figsize=(13, 8))
ax = plt.subplot(1, 2, 1)
ax.set_title("Validation Data")
ax.set_autoscaley_on(False)
ax.set_ylim([32, 43])
ax.set_autoscalex_on(False)
ax.set_xlim([-126, -112])
plt.scatter(validation_examples["longitude"],
            validation_examples["latitude"],
            cmap="coolwarm",
            c=validation_targets["median_house_value"] / validation_targets["median_house_value"].max())
ax = plt.subplot(1,2,2)
ax.set_title("Training Data")
ax.set_autoscaley_on(False)
ax.set_ylim([32, 43])
ax.set_autoscalex_on(False)
ax.set_xlim([-126, -112])
plt.scatter(training_examples["longitude"],
            training_examples["latitude"],
            cmap="coolwarm",
            c=training_targets["median_house_value"] / training_targets["median_house_value"].max())
_ = plt.plot()

# 3 Input Function:
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
     
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}                                           
 
    # Construct a dataset, and configure batching/repeating
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified
    if shuffle:
      ds = ds.shuffle(10000)
    
    # Return the next batch of data
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

# + Combine multi features:
def construct_feature_columns(input_features):
    return set([tf.feature_column.numeric_column(my_feature)
            for my_feature in input_features])

# 4 Training the Model:
def train_nn_regression_model(
    my_optimizer,
    steps,
    batch_size,
    hidden_units,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):

    # plot on another figure:
    plt.figure()
    
    periods = 10
    steps_per_period = steps / periods
  
    # 4-1) Create a dnn_regressor object.
    # my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    dnn_regressor = tf.estimator.DNNRegressor(
        feature_columns=construct_feature_columns(training_examples),
        hidden_units=hidden_units,
        optimizer=my_optimizer
    )

    # 4-2) Create input functions.
    training_input_fn = lambda:my_input_fn(training_examples,   training_targets, batch_size=batch_size)

    predict_training_input_fn = lambda: my_input_fn(training_examples,   training_targets, num_epochs=1, shuffle=False)

    predict_validation_input_fn = lambda: my_input_fn(validation_examples,   validation_targets, num_epochs=1, shuffle=False)

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("RMSE (on training data):")
    training_rmse = []
    validation_rmse = []
    for period in range (0, periods):
        # Train the model, starting from the prior state.
        dnn_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period,
        )
        # 4-3) Take a break and compute predictions.
        training_predictions = dnn_regressor.predict(input_fn=predict_training_input_fn)
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])

        validation_predictions = dnn_regressor.predict(input_fn=predict_validation_input_fn)
        validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])

        # Compute training and validation loss.
        training_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(training_predictions, training_targets))
        validation_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(validation_predictions, validation_targets))
        # Occasionally print the current loss.
        print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))
        # Add the loss metrics from this period to our list.
        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)
    print("Model training finished.")
    print("Final RMSE (on training data): %0.2f" % training_root_mean_squared_error)
    print("Final RMSE (on validation data): %0.2f" % validation_root_mean_squared_error)

    # Output a graph of loss metrics over periods.
    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(training_rmse, label="training")
    plt.plot(validation_rmse, label="validation")
    plt.legend()

    return dnn_regressor, training_rmse, validation_rmse

# 5 Improving Neural Net Performance:
# 5-1) Normalize input features on the same linear scale:
# +: Linear Scaling: [-1.0,1.0]
def linear_scale(series):
    min_val = series.min()
    max_val = series.max()
    scale = (max_val - min_val)/ 2.0
    return series.apply(lambda x:((x - min_val) / scale) - 1.0)
# +2: Additional normalization functions:
def log_normalize(series):
    return series.apply(lambda x: math.log(x+1.0))
def clip(series,clip_to_min,clip_to_max):
    return series.apply(lambda x: (min(max(x,clip_min),clip_max)))
def z_score_normalize(series):
    mean = series.mean()
    std_dv = series.std()
    return series.apply(lambda x: (x-mean)/std_dv)
def binary_threshold(series, threshold):
    return series.apply(lambda x: (1 if x > threshold else 0))
def normalize(examples_dataframe):
    processed_features = pd.DataFrame()
    processed_features["total_bedrooms"] = log_normalize(examples_dataframe["total_bedrooms"])
    processed_features["households"] = log_normalize(examples_dataframe["households"])
    processed_features["median_income"] = log_normalize(examples_dataframe["median_income"])

    processed_features["latitude"] = linear_scale(examples_dataframe["latitude"])
    processed_features["longitude"] = linear_scale(examples_dataframe["longitude"])
    processed_features["housing_median_age"] = linear_scale(examples_dataframe["housing_median_age"])

    processed_features["population"] = linear_scale(examples_dataframe["population"])
    processed_features["rooms_per_person"] = linear_scale(examples_dataframe["rooms_per_person"])
    processed_features["total_rooms"] = linear_scale(examples_dataframe["total_rooms"])
    return processed_features

normalized_training_examples = normalize(training_examples)
normalized_validation_examples = normalize(validation_examples)    
_ = normalized_training_examples.hist(bins=20, figsize=(18, 12), xlabelsize=10)  
    

# 5 Improving Neural Net Performance:
# 5-2) Try another Optimizer:
# 6 Start training & revise the hyperparameters:
dnn_regressor, training_losses, validation_losses = train_nn_regression_model(
    # my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.03),
    my_optimizer = tf.train.AdagradOptimizer(learning_rate=0.5),
    # my_optimizer = tf.train.AdamOptimizer(learning_rate=0.009),
    steps=2000,
    batch_size=5,
    hidden_units=[10,10],
    training_examples=normalized_training_examples,
    training_targets=training_targets,
    validation_examples=normalized_validation_examples, 
    validation_targets=validation_targets)


# 7 Test the model Using test dataset:
california_housing_test_data = pd.read_csv("/home/roboticslab16/code/study/Google ML Crash Course/california_housing_test.csv", sep=",")
# california_housing_test_data = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_test.csv", sep=",")
# Data Extraction:
test_examples = preprocess_features(california_housing_test_data)
normalized_test_examples = normalize(test_examples)
test_targets = preprocess_targets(california_housing_test_data)
# Input function:
predict_test_input_fn = lambda: my_input_fn(normalized_test_examples,   test_targets, num_epochs=1, shuffle=False)
# Prediction:
test_predictions = dnn_regressor.predict(input_fn=predict_test_input_fn)
test_predictions = np.array([item['predictions'][0] for item in test_predictions])
# Compute training and validation loss.
training_root_mean_squared_error = math.sqrt(
    metrics.mean_squared_error(test_predictions, test_targets))
print("Final RMSE (on test data): %0.2f" % training_root_mean_squared_error)
