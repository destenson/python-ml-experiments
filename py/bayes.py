import numpy as np
import tensorflow as tf
import tf_keras
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

# !pip install pymc3
import pymc as pm
import pandas as pd

from matplotlib import pylab as plt
# %matplotlib inline
import scipy.stats

import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA

aapl_data = get_ticker("AAPL", verbose=1)

## collect 70 samples
# interesting_columns = ['Open', 'Close', 'High', 'Low']
# interesting_columns = ['Adj Close']
interesting_columns = ['Close',]
# observed_counts = aapl_data
# print(observed_counts)

observed_counts = aapl_data.dropna(inplace=False)[interesting_columns][:70]
observed_counts = observed_counts.values.reshape((-1,))
print(observed_counts)
# observed_counts = observed_counts.values*100
# print(observed_counts)
# observed_counts = observed_counts.values*100
# observed_counts = observed_counts[['Open','Close']].values*100

plt.plot(observed_counts, label=interesting_columns),
plt.legend()
plt.plot()


# true_rates = [40, 3, 20, 50]
# true_durations = [10, 20, 5, 35]

# observed_counts = tf.concat(
#     [tfd.Poisson(rate).sample(num_steps)
#      for (rate, num_steps) in zip(true_rates, true_durations)], axis=0)

# print(observed_counts.shape)
# print(observed_counts.shape)
# # plt.plot(observed_counts)

# convert observed_counts to a double tensor instead of float

# true_rates = [40, 3, 20, 50]
# true_durations = [10, 20, 5, 35]

# observed_counts = tf.concat(
#     [tfd.Poisson(rate).sample(num_steps)
#      for (rate, num_steps) in zip(true_rates, true_durations)], axis=0)

# plt.plot(observed_counts)



# tf.compat.v1.enable_eager_execution()

num_states = 4  # e.g., bull, bear, high volatility, low volatility

initial_state_logits = tf.zeros([num_states]) # uniform distribution

daily_change_prob = 0.2
transition_probs = tf.fill([num_states, num_states],
                           daily_change_prob / (num_states - 1))
transition_probs = tf.linalg.set_diag(transition_probs,
                                      tf.fill([num_states],
                                              1 - daily_change_prob))

print("Initial state logits:\n{}".format(initial_state_logits))
print("Transition matrix:\n{}".format(transition_probs))
# print("Observed counts:\n{}".format(observed_counts))


trainable_means = tf.Variable(tf.random.normal([num_states]), name='means')
trainable_stds = tf.Variable(tf.exp(tf.random.normal([num_states])), name='stds')

hmm = tfd.HiddenMarkovModel(
    initial_distribution=tfd.Categorical(logits=initial_state_logits),
    transition_distribution=tfd.Categorical(probs=transition_probs),
    observation_distribution=tfd.Normal(loc=trainable_means, scale=trainable_stds),
    num_steps=len(observed_counts))

# Priors
mean_prior = tfd.Normal(2e2, 1e1)  # centered around 1m with large variance
std_prior = tfd.LogNormal(-3, 1)  # positive, with mode around 0.05

def log_prob():
    return (tf.reduce_sum(mean_prior.log_prob(trainable_means)) +
            tf.reduce_sum(std_prior.log_prob(trainable_stds)) +
            hmm.log_prob(observed_counts))
    

means = tf.exp(trainable_means)
print("Inferred means: {}".format(means))
# print("True rates: {}".format(true_rates))


print(observed_counts)
# Runs forward-backward algorithm to compute marginal posteriors.
posterior_dists = hmm.posterior_marginals(tf.cast(observed_counts, dtype=tf.float32))
print(posterior_dists)
posterior_probs = posterior_dists.probs_parameter()







def plot_state_posterior(ax, state_posterior_probs, title):
  ln1 = ax.plot(state_posterior_probs, c='blue', lw=3, label='p(state | price)')
  ax.set_ylim(0., 1.1)
  ax.set_ylabel('posterior probability')
  ax2 = ax.twinx()
  ln2 = ax2.plot(observed_counts, c='black', alpha=0.3, label='price')
  ax2.set_title(title)
  ax2.set_xlabel("time")
  lns = ln1+ln2
  labs = [l.get_label() for l in lns]
  ax.legend(lns, labs, loc=4)
  ax.grid(True, color='white')
  ax2.grid(False)


fig = plt.figure(figsize=(10, 10))
plot_state_posterior(fig.add_subplot(3, 2, 1),
                     posterior_probs[:, 0],
                     title="state 0 (val {:.2f})".format(means[0]))
plot_state_posterior(fig.add_subplot(3, 2, 2),
                     posterior_probs[:, 1],
                     title="state 1 (val {:.2f})".format(means[1]))
plot_state_posterior(fig.add_subplot(3, 2, 3),
                     posterior_probs[:, 2],
                     title="state 2 (val {:.2f})".format(means[2]))
plot_state_posterior(fig.add_subplot(3, 2, 4),
                     posterior_probs[:, 3],
                     title="state 3 (val {:.2f})".format(means[3]))
plot_state_posterior(fig.add_subplot(3, 2, 5),
                     posterior_probs[:, 4],
                     title="state 4 (val {:.2f})".format(means[4]))
plot_state_posterior(fig.add_subplot(3, 2, 6),
                     posterior_probs[:, 5],
                     title="state 5 (val {:.2f})".format(means[5]))
plt.tight_layout()
