

import pandas as pd
import pandas_datareader
from pandas_datareader.data import DataReader, FredReader
from datetime import datetime
import statsmodels.api as sm
import matplotlib.pyplot as plt

def interesting_fred_symbols():
    return [
        ("M2SL", "Money supply, M2, seasonally adjusted ($B)"),
        ("USREC", "US recessions (1=recession, 0=expansion)"),
        ("MRTSSM4481USN", "US Retail Clothing Sales ($M)"),
        ("TTLCON", "Total Construction, not seasonally adjusted ($M)")
        ("TTLCONS", "Total Construction, seasonally adjusted ($M)")
    ]

# Load the data
usrec = DataReader("USREC", "fred", start=datetime(1947, 1, 1), end=datetime(2024, 7, 1))

if isinstance(usrec, FredReader):
    print("Is FredReader")
    usrec = usrec.data
else:
    print(f"Is DataReader {usrec}")

# Print the first few rows of the data
print(usrec.head())

# Print the last few rows of the data
print(usrec.tail())

print(f"length:   {len(usrec)}")

# count nonzeros in the data
ct = (usrec != 0).sum()
print(f"nonzeros: {ct}")

ct = (usrec == 0).sum()
print(f"zeros:    {ct}")

# plot the data
usrec.plot()
plt.show()

# Define the Markov switching model
mod = sm.tsa.MarkovAutoregression(usrec, k_regimes=2, order=4, switching_ar=False)

# Fit the model
res = mod.fit()

# Print the summary of the model
print(res.summary())

# Plot the smoothed probabilities of the regimes
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(res.smoothed_marginal_probabilities[0], label='Regime 1')
ax.plot(res.smoothed_marginal_probabilities[1], label='Regime 2')
ax.legend()
plt.show()


# tensorflow also has some data available

