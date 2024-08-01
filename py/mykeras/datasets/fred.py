import pandas as pd
import pandas_datareader
from pandas_datareader.data import DataReader, FredReader
from datetime import datetime
import statsmodels.api as sm
import matplotlib.pyplot as plt
import tensorflow as tf

def interesting_fred_symbols():
    return [
        ("M2SL", "Money supply, M2, seasonally adjusted ($B)"),
        ("USREC", "US recessions (1=recession, 0=expansion)"),
        ("MRTSSM4481USN", "US Retail Clothing Sales ($M)"),
        ("TTLCON", "Total Construction, not seasonally adjusted ($M)"),
        ("TTLCONS", "Total Construction, seasonally adjusted ($M)"),
    ]

def get_interesting_data(start=datetime(1947, 1, 1), end=datetime(2024, 7, 1)):
    data = {}
    for symbol, desc in interesting_fred_symbols():
        data[symbol] = {
            'desc': desc,
            'data': DataReader(symbol, "fred", start=start, end=end),
        }

    return data
    
    
def show_data(df):
    if isinstance(df, FredReader):
        print("Is FredReader")
        df = df.data
    elif isinstance(df, tuple):
        df = df['data']
    elif isinstance(df, pd.DataFrame):
        print(f"Is DataReader {df}")

    # Print the first few rows of the data
    print(df.head())

    # Print the last few rows of the data
    print(df.tail())

    print(f"length:   {len(df)}")

    # count nonzeros in the data
    ct = (df != 0).sum()
    print(f"nonzeros: {ct}")

    ct = (df == 0).sum()
    print(f"zeros:    {ct}")

    # plot the data
    df.plot()
    plt.show()


class FredTest(tf.test.TestCase):
    def test_show_data(self):
        df = DataReader("USREC", "fred", start=datetime(1947, 1, 1), end=datetime(2024, 7, 1))
        show_data(df)
        # raise ValueError("Done")
        
    def test_get_interesting_data(self):
        data = get_interesting_data()
        for symbol, desc in interesting_fred_symbols():
            print(f"{symbol}: {desc}")
            show_data(data[symbol]['data'])
        # raise ValueError("Done")

if __name__ == "__main__":
    tf.test.main()
