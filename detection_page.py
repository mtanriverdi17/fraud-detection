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
st.markdown("* Do these distributions make sense? ",
            "* Yes. You've normalized the input and these are mostly concentrated in the `+/- 2` range.")
st.markdown("* Can you see the difference between the distributions?",
            "* Yes the positive examples contain a much higher rate of extreme values.")

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


loaded_model = keras.models.load_model('my_model')
loaded_model.predict(train_features[0:1])


st.pyplot(positive_distribution)
st.pyplot(positive_distribution)
