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
Titanic_dataframe = pd.read_csv("/home/roboticslab16/code/study/Google ML Crash Course/Kaggle/train_age_revised.csv", sep=",")
Titanic_dataframe.describe()
Titanic_dataframe = Titanic_dataframe.reindex(
    np.random.permutation(Titanic_dataframe.index))


# 1 Data preprocess: features & targets
def preprocess_features(Titanic_dataframe):
    selected_features = Titanic_dataframe[
    [
        # "PassengerId",
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked"
        ]]
    processed_features = selected_features.copy()
    processed_features["Sex"] = processed_features["Sex"].apply(lambda x: 1 if x == "male" else 0)
    processed_features["Embarked"] = processed_features["Embarked"].apply(lambda x: -1 if x == "S" else 0 if x == "S" else 1)
    
    # Create a synthetic feature.
    # processed_features["rooms_per_person"] = (
    # Titanic_dataframe["total_rooms"] /
    # Titanic_dataframe["population"])
    return processed_features

def preprocess_targets(Titanic_dataframe):
    output_targets = pd.DataFrame()
    # Scale the target to be in units of thousands of dollars.
    output_targets["Survived"] = (Titanic_dataframe["Survived"] > 0.5).astype(float)
    
    return output_targets

# 2 Extract useful data:
training_examples = preprocess_features(Titanic_dataframe.head(890))
training_examples.describe()
training_targets = preprocess_targets(Titanic_dataframe.head(890))
training_targets.describe()
validation_examples = preprocess_features(Titanic_dataframe.tail(300))
validation_examples.describe()
validation_targets = preprocess_targets(Titanic_dataframe.tail(300))
validation_targets.describe()

# 3 Input Function:
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}                                           
 
    # Construct a dataset, and configure batching/repeating
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # # Shuffle the data, if specified
    # if shuffle:
    #   ds = ds.shuffle(10000)
    
    # Return the next batch of data
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

# + Combine multi features:
def construct_feature_columns(input_features):
    return set([tf.feature_column.numeric_column(my_feature)
            for my_feature in input_features])

# 4 Training the Model:
def train_model(
    learning_rate,
    steps,
    batch_size,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):

    periods = 10
    steps_per_period = steps / periods

        # 4-1) Create a linear classifier object.
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_classifier = tf.estimator.LinearClassifier(
        feature_columns=construct_feature_columns(training_examples),
        optimizer=my_optimizer
    )

    # 4-2) Create input functions.    
    training_input_fn = lambda:my_input_fn(training_examples,   training_targets, batch_size=batch_size)

    predict_training_input_fn = lambda: my_input_fn(training_examples,   training_targets, num_epochs=1, shuffle=False)

    predict_validation_input_fn = lambda: my_input_fn(validation_examples,   validation_targets, num_epochs=1, shuffle=False)

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("LogLoss (on training data):")
    training_logloss_all = []
    validation_log_loss_all = []
    for period in range (0, periods):
        # Train the model, starting from the prior state.
        linear_classifier.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )
        # Take a break and compute predictions.    
        training_predictions = linear_classifier.predict(input_fn=predict_training_input_fn)
        training_predictions = np.array([item['probabilities'] for item in training_predictions])

        validation_predictions = linear_classifier.predict(input_fn=predict_validation_input_fn)
        validation_predictions = np.array([item['probabilities'] for item in validation_predictions])

        training_log_loss = metrics.log_loss(training_targets, training_predictions)
        validation_log_loss = metrics.log_loss(validation_targets, validation_predictions)
        # Occasionally print the current loss.
        print("  period %02d : %0.2f" % (period, training_log_loss))
        # Add the loss metrics from this period to our list.
        training_logloss_all.append(training_log_loss)
        validation_log_loss_all.append(validation_log_loss)
    print("Model training finished.")
    # Output a graph of loss metrics over periods.
    plt.ylabel("LogLoss")
    plt.xlabel("Periods")
    plt.title("LogLoss vs. Periods")
    plt.tight_layout()
    plt.plot(training_logloss_all, label="training")
    plt.plot(training_logloss_all, label="validation")
    plt.legend()

    return linear_classifier 

# 5 Start training & revise the hyperparameters:
# Don't forget to adjust these parameters while using different features!
linear_classifier = train_model(
    learning_rate=0.00003,
    steps=100,
    batch_size=5,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)  