import tensorflow as tf


def study_data(data, window_size=69, num_states=4, num_runs=10, daily_change_prob=0.2):
    print(f"study_data({locals()})")

    # e.g., bull, bear, high volatility, low volatility
    num_states = 4 if not num_states > 0 else num_states

    initial_state_logits = tf.zeros([num_states]) # uniform distribution

    daily_change_prob = 0.2 if not daily_change_prob > 0.0 else daily_change_prob

    print(f"initial_state_logits: {initial_state_logits}")
    print(f"daily_change_prob: {daily_change_prob}")

    transition_probs = tf.fill([num_states, num_states],
                            daily_change_prob / (num_states - 1))
    transition_probs = tf.linalg.set_diag(transition_probs,
                                        tf.fill([num_states],
                                                1 - daily_change_prob))

    print("Initial state logits:\n{}".format(initial_state_logits))
    print("Transition matrix:\n{}".format(transition_probs))
    # print("Observed counts:\n{}".format(observations))


    trainable_means = tf.Variable(tf.random.normal([num_states]), name='means')
    trainable_stds = tf.Variable(tf.exp(tf.random.normal([num_states])), name='stds')

    hmm = tfd.HiddenMarkovModel(
        initial_distribution=tfd.Categorical(logits=initial_state_logits),
        transition_distribution=tfd.Categorical(probs=transition_probs),
        observation_distribution=tfd.Normal(loc=trainable_means, scale=trainable_stds),
        num_steps=len(observations))

    # Priors
    mean_prior = tfd.Normal(2e2, 1e1)  # centered around 1m with large variance
    std_prior = tfd.LogNormal(-3, 1)  # positive, with mode around 0.05


    means = tf.exp(trainable_means)
    print("Inferred means: {}".format(means))
    # print("True rates: {}".format(true_rates))


    print(observations)
    # Runs forward-backward algorithm to compute marginal posteriors.
    posterior_dists = hmm.posterior_marginals(tf.cast(observations, dtype=tf.float32))
    print(posterior_dists)
    posterior_probs = posterior_dists.probs_parameter()

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






def plot_state_posterior(ax, state_posterior_probs, title):
    ln1 = ax.plot(state_posterior_probs, c='blue', lw=3, label='p(state | price)')
    ax.set_ylim(0., 1.1)
    ax.set_ylabel('posterior probability')
    ax2 = ax.twinx()
    ln2 = ax2.plot(observations, c='black', alpha=0.3, label='price')
    ax2.set_title(title)
    ax2.set_xlabel("time")
    lns = ln1+ln2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=4)
    ax.grid(True, color='white')
    ax2.grid(False)


def log_prob(hmm, mean_prior, std_prior, means, stds, observations):
    return (tf.reduce_sum(mean_prior.log_prob(means)) +
            tf.reduce_sum(std_prior.log_prob(stds)) +
            hmm.log_prob(observations))
    



aapl_data = get_ticker("AAPL", verbose=1, nocache=True)

## collect 70 samples
# interesting_columns = ['Open', 'Close', 'High', 'Low']
# interesting_columns = ['Adj Close']
interesting_columns = ['Close',]
# observations = aapl_data
# print(observations)

observations = aapl_data.dropna(inplace=False)[interesting_columns][:70]
observations = observations.values.reshape((-1,))
print(observations)
# observations = observations.values*100
# print(observations)
# observations = observations.values*100
# observations = observations[['Open','Close']].values*100

plt.plot(observations, label=interesting_columns),
plt.legend()
plt.plot()


# true_rates = [40, 3, 20, 50]
# true_durations = [10, 20, 5, 35]

# observations = tf.concat(
#     [tfd.Poisson(rate).sample(num_steps)
#      for (rate, num_steps) in zip(true_rates, true_durations)], axis=0)

# print(observations.shape)
# print(observations.shape)
# # plt.plot(observations)

# convert observations to a double tensor instead of float

# true_rates = [40, 3, 20, 50]
# true_durations = [10, 20, 5, 35]

# observations = tf.concat(
#     [tfd.Poisson(rate).sample(num_steps)
#      for (rate, num_steps) in zip(true_rates, true_durations)], axis=0)

# plt.plot(observations)



# tf.compat.v1.enable_eager_execution()
