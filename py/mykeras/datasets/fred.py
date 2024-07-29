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

def get_interesting_data(start=datetime(1947, 1, 1), end=datetime(2024, 7, 1)):
    data = {}
    for symbol, desc in interesting_fred_symbols():
        data[symbol].desc = desc
        data[symbol].data = DataReader(symbol, "fred", start=start, end=end)
        
    return data
    
    
def show_data(df):
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
