# Ref: 8_feature_corsses.py
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
# Reindex the data:
# california_housing_dataframe = california_housing_dataframe.reindex(
#     np.random.permutation(california_housing_dataframe.index))


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
    # Create a boolean categorical feature representing whether the
    # median_house_value is above a set threshold.
    output_targets["median_house_value_is_high"] = (
        california_housing_dataframe["median_house_value"] > 265000).astype(float)
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


# 4 Training the Model:
def train_model(
    learning_rate,
    regularization_strength,
    steps,
    batch_size,
    feature_columns,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):

    periods = 10
    steps_per_period = steps / periods

    # 4-1) Create a linear regressor object(GradientDescentOptimizer -> FtrlOptimizer).
    my_optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate,l1_regularization_strength=regularization_strength)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_classfier = tf.estimator.LinearClassifier(
        feature_columns=feature_columns,
        optimizer=my_optimizer
    )

    # 4-2) Create input functions.
    training_input_fn = lambda:my_input_fn(training_examples,   training_targets, batch_size=batch_size)

    predict_training_input_fn = lambda: my_input_fn(training_examples,   training_targets, num_epochs=1, shuffle=False)

    predict_validation_input_fn = lambda: my_input_fn(validation_examples,   validation_targets, num_epochs=1, shuffle=False)

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("LogLoss (on validation data):")
    training_log_losses = []
    validation_log_losses = []
    for period in range (0, periods):
    # Train the model, starting from the prior state.
        linear_classfier.train(
            input_fn=training_input_fn,
            steps=steps_per_period,
        )
        # 4-3) Take a break and compute predictions.
        training_predictions = linear_classfier.predict(input_fn=predict_training_input_fn)
        training_predictions = np.array([item['probabilities'] for item in training_predictions])

        validation_predictions = linear_classfier.predict(input_fn=predict_validation_input_fn)
        validation_predictions = np.array([item['probabilities']for item in validation_predictions])

        # Compute training and validation loss.
        training_log_loss = metrics.log_loss(training_targets, training_predictions)
        validation_log_loss = metrics.log_loss(validation_targets, validation_predictions)
        # Occasionally print the current loss.
        print("  period %02d : %0.2f" % (period, validation_log_loss))
        # Add the loss metrics from this period to our list.
        training_log_losses.append(training_log_loss)
        validation_log_losses.append(validation_log_loss)
    print("Model training finished.")

    # Output a graph of loss metrics over periods.
    plt.ylabel("LogLoss")
    plt.xlabel("Periods")
    plt.title("LogLoss vs. Periods")
    plt.tight_layout()
    plt.plot(training_log_losses, label="training")
    plt.plot(validation_log_losses, label="validation")
    plt.legend()

    return linear_classfier

# +1 Bucketize the features!
# def get_quantile_based_boundaries(feature_values, num_buckets):
#     boundaries = np.arange(1.0, num_buckets) / num_buckets
#     quantiles = feature_values.quantile(boundaries)
#     return [quantiles[q] for q in quantiles.keys()]
def get_quantile_based_buckets(feature_values, num_buckets):
    quantiles = feature_values.quantile(
        [(i+1.)/(num_buckets + 1.) for i in range(num_buckets)])
    return [quantiles[q] for q in quantiles.keys()]    

# +2 Combine multi features:
def construct_feature_columns():
    households = tf.feature_column.numeric_column("households")
    longitude = tf.feature_column.numeric_column("longitude")
    latitude = tf.feature_column.numeric_column("latitude")
    housing_median_age = tf.feature_column.numeric_column("housing_median_age")
    median_income = tf.feature_column.numeric_column("median_income")
    rooms_per_person = tf.feature_column.numeric_column("rooms_per_person")

    # # Divide households into 7 buckets.
    # bucketized_households = tf.feature_column.bucketized_column(
    # households, boundaries=get_quantile_based_boundaries(
    #     training_examples["households"], 7))

    # # Divide longitude into 10 buckets.
    # bucketized_longitude = tf.feature_column.bucketized_column(
    # longitude, boundaries=get_quantile_based_boundaries(
    #     training_examples["longitude"], 10))

    # bucketized_latitude = tf.feature_column.bucketized_column(
    # latitude, boundaries=get_quantile_based_boundaries(
    #     training_examples["latitude"], 7))

    # bucketized_housing_median_age = tf.feature_column.bucketized_column(
    # housing_median_age, boundaries=get_quantile_based_boundaries(
    #     training_examples["housing_median_age"], 7))

    # bucketized_median_income = tf.feature_column.bucketized_column(
    # median_income, boundaries=get_quantile_based_boundaries(
    #     training_examples["median_income"], 7))

    # Divide households into 7 buckets.
    bucketized_households = tf.feature_column.bucketized_column(
    households, boundaries=get_quantile_based_buckets(
        training_examples["households"], 7))

    # Divide longitude into 10 buckets.
    bucketized_longitude = tf.feature_column.bucketized_column(
    longitude, boundaries=get_quantile_based_buckets(
        training_examples["longitude"], 10))

    bucketized_latitude = tf.feature_column.bucketized_column(
    latitude, boundaries=get_quantile_based_buckets(
        training_examples["latitude"], 7))

    bucketized_housing_median_age = tf.feature_column.bucketized_column(
    housing_median_age, boundaries=get_quantile_based_buckets(
        training_examples["housing_median_age"], 7))

    bucketized_median_income = tf.feature_column.bucketized_column(
    median_income, boundaries=get_quantile_based_buckets(
        training_examples["median_income"], 7))

    # +3 Make a feature colunm for feature corss: longitude and latitude cross
    # crossed_column() usage: tf.feature_column.crossed_column(keys,hash_bucket_size,hash_key=None)
    long_x_lat = tf.feature_column.crossed_column([bucketized_longitude, bucketized_latitude],1000)

    feature_columns = set([
    bucketized_longitude,
    bucketized_latitude,
    bucketized_housing_median_age,
    bucketized_households,
    bucketized_median_income,
    long_x_lat])

    return feature_columns

# 5 Run the model training:
linear_classifier = train_model(
    learning_rate=1.0,
    regularization_strength=0.0,
    steps=500,
    batch_size=100,
    feature_columns=construct_feature_columns(),
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

# 6 Calculate the model size
def model_size(estimator):
    variables = estimator.get_variable_names()
    size = 0
    for variable in variables:
        if not any(x in variable 
            for x in ['global_step',
                        'centered_bias_weight',
                        'bias_weight',
                        'Ftrl']
                    ):
            size += np.count_nonzero(estimator.get_variable_value(variable))
    return size
print("Model size:", model_size(linear_classifier))
