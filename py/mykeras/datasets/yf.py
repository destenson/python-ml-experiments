import tensorflow as tf
import yfinance as yf
import numpy as np
import pandas as pd

from datetime import date

from yfinance import Ticker

from requests import Session
from requests_cache import CacheMixin, SQLiteCache
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket
from pyrate_limiter import Duration, RequestRate, Limiter

# import requests_cache

class CachedLimiterSession(CacheMixin, LimiterMixin, Session):
    pass

session = CachedLimiterSession(
    limiter=Limiter(RequestRate(2, Duration.SECOND*5)),  # max 2 requests per 5 seconds
    bucket_class=MemoryQueueBucket,
    backend=SQLiteCache("data/yfinance.cache"),
)

# load data
def yahoofinance_data(symbol, start='2023-01-01', end='2024-01-01', verbose=False):
    dataset = get_ticker_data(symbol, start=start, end=end)
    
    if verbose:
        print(f"dataset: {dataset}")
    
    return dataset

def get_tickers(symbols, start='2023-01-01', end='2024-01-01', verbose=False):
    datasets = []
    for symbol in symbols:
        dataset = get_ticker_data(symbol, start=start, end=end)
        datasets.append(dataset)
        print(f"dataset: {dataset}") if verbose > 0 else None
    return np.array(datasets)

def get_ticker_data(ticker_symbol,
                    start="2023-01-01", end=date.today(),
                    period="1d", cache_dir='data/'):
    cache_filename = f"{cache_dir}{ticker_symbol}_{start}_{end}.pickle"
    try:
        import os
        if cache_filename and os.stat(cache_filename):
            print(f"Loading data from cache: {cache_filename}")
            return pd.read_pickle(cache_filename)
    except Exception as e:
        print(f"caught exception: {e}")

    print("Getting ticker data for {ticker_symbol}")
    # create Ticker object    
    # symbol = Ticker(ticker_symbol)
    symbol = Ticker(ticker_symbol, session=session)

    # # get all stock info
    # symbol.info

    # get historical market data
    hist = symbol.history(period=period, start=start, end=end)

    # # show meta information about the history (requires history() to be called first)
    # symbol.history_metadata

    # # show actions (dividends, splits, capital gains)
    # symbol.actions
    # symbol.dividends
    # symbol.splits
    # symbol.capital_gains  # only for mutual funds & etfs

    # # show share count
    # symbol.get_shares_full(start=start, end=end)

    # # show financials:
    # # - income statement
    # symbol.income_stmt
    # symbol.quarterly_income_stmt
    # # - balance sheet
    # symbol.balance_sheet
    # symbol.quarterly_balance_sheet
    # # - cash flow statement
    # symbol.cashflow
    # symbol.quarterly_cashflow
    # # see `Ticker.get_income_stmt()` for more options

    # try:
    #     # show holders
    #     symbol.major_holders
    #     symbol.institutional_holders
    #     symbol.mutualfund_holders
    #     symbol.insider_transactions
    #     symbol.insider_purchases
    #     symbol.insider_roster_holders
    # except Exception as e:
    #     print(f"caught exception (ticker={ticker_symbol}): {e}")

    # symbol.sustainability

    # # show recommendations
    # symbol.recommendations
    # symbol.recommendations_summary
    # symbol.upgrades_downgrades

    # # Show future and historic earnings dates, returns at most next 4 quarters and last 8 quarters by default.
    # # Note: If more are needed use msft.get_earnings_dates(limit=XX) with increased limit argument.
    # symbol.earnings_dates

    # # show ISIN code - *experimental*
    # # ISIN = International Securities Identification Number
    # symbol.isin

    # # show options expirations
    # symbol.options

    # # show news
    # symbol.news

    # if option_expiration_dates := symbol.options:
    #     print(f"option expiration dates: {option_expiration_dates}")
        
    #     # get option chain for specific expiration
    #     opt = [symbol.option_chain(d) for d in option_expiration_dates[:1]]
    #     # data available via: opt.calls, opt.puts
    #     # print(f"option chain for {option_expiration_dates[0]}: {opt}")
                
    # symbol.history(..., proxy="PROXY_SERVER")
    # symbol.get_actions(proxy="PROXY_SERVER")
    # symbol.get_dividends(proxy="PROXY_SERVER")
    # symbol.get_splits(proxy="PROXY_SERVER")
    # symbol.get_capital_gains(proxy="PROXY_SERVER")
    # symbol.get_balance_sheet(proxy="PROXY_SERVER")
    # symbol.get_cashflow(proxy="PROXY_SERVER")
    # symbol.option_chain(..., proxy="PROXY_SERVER")

    # from deepcopy import deepcopy 
    result = np.array({
        'symbol': ticker_symbol,
        'data': {
            'actions': symbol.get_actions(),
            # 'analyst_price_target': symbol.get_analyst_price_target(as_dict=True), # not implemented
            'balance_sheet': symbol.get_balance_sheet(as_dict=True),
            'calendar': symbol.get_calendar(),
            'capital_gains': symbol.get_capital_gains(),
            'cash_flow': symbol.get_cash_flow(as_dict=True),
            'dividends': symbol.get_dividends(),
            'earnings_dates': symbol.get_earnings_dates(),
            # 'earnings_forecast': symbol.get_earnings_forecast(as_dict=True), # not implemented
            # 'earnings_trend': symbol.get_earnings_trend(as_dict=True), # not implemented
            'financials': symbol.get_financials(as_dict=True),
            'income_stmt': symbol.get_income_stmt(as_dict=True),
            'insider_purchases': symbol.get_insider_purchases(as_dict=True),
            'insider_roster_holders': symbol.get_insider_roster_holders(as_dict=True),
            'insider_transactions': symbol.get_insider_transactions(as_dict=True),
            'institutional_holders': symbol.get_institutional_holders(as_dict=True),
            'major_holders': symbol.get_major_holders(as_dict=True),
            'news': symbol.get_news(),
            'recommendations': symbol.get_recommendations(as_dict=True),
            'recommendations_summary': symbol.get_recommendations_summary(as_dict=True),
            # 'rev_forecast': symbol.get_rev_forecast(as_dict=True), # not implemented
            # 'shares': symbol.get_shares(as_dict=True), # not implemented
            'shares_full': symbol.get_shares_full(start=start, end=end),
            'splits': symbol.get_splits(),
            # 'trend_details': symbol.get_trend_details(as_dict=True), # not implemented
            'upgrades_downgrades': symbol.get_upgrades_downgrades(as_dict=True),
        },
        'history': hist
    })
    
    import pickle
    pickle.dump(result, open(cache_filename, 'wb'))
    # result.to_pickle(cache_filename)
    return result
    
    # try:
    #     data = yf.Ticker(symbol)
    #     # data = yf.download(symbol, start=start, end=end)
    #     print(f"Downloaded data for {symbol}")
    #     print(f"data.shape: {data.shape}")
    #     print(f"data.T: {data.T}")
    #     print(f"data.describe: {data.describe()}")
    #     print(f"data.head(): {data.head()}")
    #     data = (
    #         data[:]
    #         .rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"})
    #     )
    #     # hist = data.history(start=start, end=end, interval="1d")
    #     # data = data[:,~data.isna()]
    #     diff = data.diff()
    #     print(f"diffs: {diff.head()}")
    #     data = data.reset_index()
    #     print(f"data: {diff.head()}")
    #     # add returns column to data
    #     data['returns_1d'] = diff['Adj Close']
    #     # data['returns_1d'] = diff.values
    #     print(f"data with returns: {diff.head()}")
    #     data.Date = data.Date.apply(lambda d: d.date())
    #     print(f"returning data: {diff.head()}")
    #     return data.dropna()
    # except Exception as e:
    #     print(f"caught exception: {e}")
    #     return None

# def test_get_ticker_data():
#     print("Getting ticker")
#     hist = get_ticker_data("AAPL")
#     print(f"Got history: {hist}")
#     print(hist.head())
#     assert hist is not None
#     assert len(hist.values) > 0
#     print(hist.columns)
#     assert hist.columns.tolist() == ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Price change']
#     assert hist.Date.dtype == np.datetime64
#     assert hist.Open.dtype == np.float64
#     assert hist.High.dtype == np.float64
#     assert hist.Low.dtype == np.float64
#     assert hist.Close.dtype == np.float64
#     assert hist.Volume.dtype == np.float64
#     assert hist['price change'].dtype == np.float64
#     assert hist.shape[0] > 0
#     assert hist.shape[1] == 7

# test_get_ticker_data()


class YahooFinanceDataTest(tf.test.TestCase):
    # def test_yf(self):
    #     data = yahoofinance_data('AAPL', verbose=True)
    #     print(f"yahoofinance data: {data}")
        
    def test_yf_multi(self):
        data = get_tickers(['AAPL', 'AMZN', 'MSFT', 'MSTR'])
        print(f"ticker data: {len(data)}")
        # summarize(data)
        


if __name__ == "__main__":
    tf.test.main()
