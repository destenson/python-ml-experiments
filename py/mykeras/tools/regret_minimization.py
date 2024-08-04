
import tensorflow as tf
import keras
from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense, InputLayer, Reshape
import numpy as np



# 1. Problem formulation:
# Define the predictive task clearly and unambiguously. 
# For instance, you might want to predict future values of a stock portfolio
# based on historical data and external factors. The goal is to make
# decisions (e.g., buy/sell/hold) that minimize regret over time.

# 2. Data preparation:
# Prepare your multidimensional time-series data. This includes:
# Feature Selection: Identify relevant features for the prediction task.
# Normalization: Scale features to ensure they are on a comparable scale.
# Segmentation: Divide the data into training, validation, and test sets.

# 3. Model selection:
# Choose a model that can capture the dynamics of the time series.
# For instance, LSTM: Effective for capturing long-term dependencies in
# sequential data.
# Transformer: Useful for handling complex dependencies and parallelizing
# computations.

# 4. Incorporate counterfactual regret minimization into the decision-making
# process of your model:
# - Action Space Definition: Define the set of possible actions.
# For example, in a stock trading scenario, actions could be buy, sell, or hold.
# - Regret Calculation: At each time step, calculate the regret for not having
# taken each possible action. Regret is typically measured as the difference
# between the actual outcome and the best possible outcome had a different
# action been taken.
# - Policy Update: Update the policy to minimize cumulative regret over time.
# This involves adjusting the action probabilities based on past regrets.

# 5. Training process:
# Initialization: Start with an initial policy (e.g., random or heuristic-based).
# Iterative Training: Train the time-series model and update the policy
# iteratively. At each iteration:
# - Predict Future Values: Use the time-series model to predict future values.
# - Choose Actions: Select actions based on the current policy.
# - Observe Outcomes: Record the actual outcomes and calculate regrets.
# - Update Policy: Adjust the policy to minimize regret.

# 6. Evaluation:
# Performance Metrics: Evaluate the model using appropriate metrics such as
# Mean Squared Error (MSE) for predictions and cumulative regret for decision-
# making.
# Backtesting: In financial applications, backtest the strategy on historical
# data to assess its performance.

def formulate_problem(data, n_actions, input_shape=None, output_shape=None):
    return {
        'input_shape': input_shape,
        'output_shape': output_shape,
        'n_actions': n_actions,
    }

def prepare_dataset(data, form):
    input_shape = form['input_shape']
    output_shape = form['output_shape']
    n_actions = form['n_actions']
    
    # TODO: feature selection, normalization, segmentation
    pass

def select_model(form, kind='lstm'):
    input_shape = form['input_shape']
    output_shape = form['output_shape']
    n_actions = form['n_actions']

    if kind == 'lstm':
        model = Sequential([
            InputLayer(shape=input_shape),
            LSTM(50),
            Dense(output_shape[0]*n_actions),
            Reshape((output_shape[0], n_actions)),
        ])
        model.compile(optimizer='adam', loss='mse')    
    elif kind == 'transformer':
        # TODO: implement transformer model
        raise ValueError("Transformer model not implemented yet.")
    else:
        raise ValueError(f"Unknown model type: {kind}")
    
    return model

def action_space():
    return [
        'hold',
        'buy at open', 'sell at open',
        'buy at close', 'sell at close',
        # 'buy market', 'sell market',
        # 'buy limit', 'sell limit',
        # 'buy stop', 'sell stop',
        # 'buy stop limit', 'sell stop limit',
        # 'buy trailing stop', 'sell trailing stop',
        # 'buy trailing stop limit', 'sell trailing stop limit',
    ]

def test_regmin():
    
    # Sample data
    data = np.random.rand(100, 10, 7) - 0.5 # 100 time steps, 10 symbols, 7 features
    actions = action_space()
    n_actions = len(actions)

    # Prepare data
    X = data[:-1]
    y = data[1:]

    # LSTM model
    model = Sequential([
        InputLayer(shape=(X.shape[1], X.shape[2])),
        LSTM(50),
        Dense(y.shape[-1]*n_actions),
        Reshape((y.shape[-1], n_actions)),
    ])
    model.compile(optimizer='adam', loss='mse')

    # CFR variables
    regret_sum = np.zeros(n_actions)
    strategy = np.ones(n_actions) / n_actions

    # Training loop
    for epoch in range(100):
        # Predict future values
        predictions = model.predict(X)
        
        # Choose actions based on current strategy
        action_probs = strategy / strategy.sum()
        chosen_actions = np.random.choice(actions, size=len(predictions), p=action_probs)
        
        # Calculate regrets
        actual_outcomes = y  # Simplified for illustration
        regrets = np.zeros((len(predictions), n_actions))
        for i, action in enumerate(actions):
            # Simplified regret calculation
            regrets[:, i] = np.abs(actual_outcomes - predictions)
        
        # Update regret sums
        regret_sum += regrets.mean(axis=0)
        
        # Update strategy
        strategy = np.maximum(regret_sum, 0)
        strategy /= strategy.sum()

        # Train model
        model.fit(X, y, epochs=1, verbose=0)

    print("Training complete.")

test_regmin()