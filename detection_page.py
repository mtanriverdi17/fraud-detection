import tensorflow as tf
from tensorflow import keras

import os
import tempfile

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st
BATCH_SIZE = 2048

st.title('Web Based Fraud Detection')

mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

#file = tf.keras.utils
raw_df = pd.read_csv(
    'https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv')
st.write('Below is the head of our dataset.', raw_df.head())


neg, pos = np.bincount(raw_df['Class'])
total = neg + pos
""" print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(total, pos, 100 * pos / total)) """
st.write('Examples\n Total: ', total, '\nPositive: ',
         pos, ' ', 100*pos/total, '% OF TOTAL')

cleaned_df = raw_df.copy()

# You don't want the `Time` column.
cleaned_df.pop('Time')

# The `Amount` column covers a huge range. Convert to log-space.
eps = 0.001  # 0 => 0.1¢
cleaned_df['Log Ammount'] = np.log(cleaned_df.pop('Amount')+eps)

# Use a utility from sklearn to split and shuffle your dataset.
train_df, test_df = train_test_split(cleaned_df, test_size=0.2)
train_df, val_df = train_test_split(train_df, test_size=0.2)

# Form np arrays of labels and features.
train_labels = np.array(train_df.pop('Class'))
bool_train_labels = train_labels != 0
val_labels = np.array(val_df.pop('Class'))
test_labels = np.array(test_df.pop('Class'))

train_features = np.array(train_df)
val_features = np.array(val_df)
test_features = np.array(test_df)


cleaned_df = raw_df.copy()

# You don't want the `Time` column.
cleaned_df.pop('Time')

# The `Amount` column covers a huge range. Convert to log-space.
eps = 0.001  # 0 => 0.1¢
cleaned_df['Log Ammount'] = np.log(cleaned_df.pop('Amount')+eps)


# Use a utility from sklearn to split and shuffle your dataset.
train_df, test_df = train_test_split(cleaned_df, test_size=0.2)
train_df, val_df = train_test_split(train_df, test_size=0.2)

# Form np arrays of labels and features.
train_labels = np.array(train_df.pop('Class'))
bool_train_labels = train_labels != 0
val_labels = np.array(val_df.pop('Class'))
test_labels = np.array(test_df.pop('Class'))

train_features = np.array(train_df)
val_features = np.array(val_df)
test_features = np.array(test_df)

scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)

val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)

train_features = np.clip(train_features, -5, 5)
val_features = np.clip(val_features, -5, 5)
test_features = np.clip(test_features, -5, 5)


st.write('Training labels shape:', train_labels.shape)
st.write('Validation labels shape:', val_labels.shape)
st.write('Test labels shape:', test_labels.shape)

st.write('Training features shape:', train_features.shape)
st.write('Validation features shape:', val_features.shape)
st.write('Test features shape:', test_features.shape)

st.markdown("### Look at the data distribution")
st.markdown('Next compare the distributions of the positive and negative examples over a few features. Good questions to ask yourself at this point are:')
st.markdown("""* Do these distributions make sense? 
            * Yes. You've normalized the input and these are mostly concentrated in the `+/- 2` range.""")
st.markdown("""* Can you see the difference between the distributions?
            * Yes the positive examples contain a much higher rate of extreme values.""")

pos_df = pd.DataFrame(
    train_features[bool_train_labels], columns=train_df.columns)
neg_df = pd.DataFrame(
    train_features[~bool_train_labels], columns=train_df.columns)

positive_distribution = sns.jointplot(pos_df['V5'], pos_df['V6'],
                                      kind='hex', xlim=(-5, 5), ylim=(-5, 5))
plt.suptitle("Positive distribution")

negative_distribution = sns.jointplot(neg_df['V5'], neg_df['V6'],
                                      kind='hex', xlim=(-5, 5), ylim=(-5, 5))
fig = plt.suptitle("Negative distribution")

st.pyplot(positive_distribution)
st.pyplot(negative_distribution)

loaded_model = keras.models.load_model('my_model')
loaded_model.predict(train_features[0:10])


results = loaded_model.evaluate(
    train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)
st.write('Loss: ', results[0])
st.markdown("""
The correct bias to set can be derived from:

$$ p_0 = pos/(pos + neg) = 1/(1+e^{-b_0}) $$
$$ b_0 = -log_e(1/p_0 - 1) $$
$$ b_0 = log_e(pos/neg)$$ """)

initial_bias = np.log([pos/neg])
st.write('Initial Bias:', initial_bias)

ib_model = keras.models.load_model('ib_model')
ib_model.predict(train_features[0:10])

st.markdown("""
With this initialization the initial loss should be approximately:

$$-p_0log(p_0)-(1-p_0)log(1-p_0) = 0.01317$$
 """)

ib_results = ib_model.evaluate(
    train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)
st.write('Loss: ', ib_results[0])

train_predictions_baseline = ib_model.predict(
    train_features, batch_size=BATCH_SIZE)
test_predictions_baseline = ib_model.predict(
    test_features, batch_size=BATCH_SIZE)


def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    fig, ax = plt.subplots()
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    st.write(fig)

    st.write('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
    st.write(
        'Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
    st.write('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
    st.write('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
    st.write('Total Fraudulent Transactions: ', np.sum(cm[1]))


baseline_results = ib_model.evaluate(test_features, test_labels,
                                     batch_size=BATCH_SIZE, verbose=0)

plot_cm(test_labels, test_predictions_baseline)

st.markdown("""
### Plot the ROC

Now plot the [ROC](https://developers.google.com/machine-learning/glossary#ROC). This plot is useful because it shows, at a glance, the range of performance the model can reach just by tuning the output threshold.
 """)


def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)
    fig, ax = plt.subplots()

    plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5, 20])
    plt.ylim([80, 100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
    st.pyplot(fig)


plot_roc("Train Baseline", train_labels,
         train_predictions_baseline, color=colors[0])
plot_roc("Test Baseline", test_labels, test_predictions_baseline,
         color=colors[0], linestyle='--')
