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
    limiter=Limiter(RequestRate(2, Duration.SECOND*7)),  # max 2 requests per 7 seconds
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
                    period="max",
                    cache_dir='data/',
                    verbose=False):
    if isinstance(cache_dir, str):
        cache_filename = f"{cache_dir}{ticker_symbol}_{start}_{end}.pickle"
        try:
            import os
            if cache_filename and os.path.isfile(cache_filename):
                print(f"Loading data from cache: {cache_filename}") if verbose > 0 else None
                return pd.read_pickle(cache_filename)
        except Exception as e:
            print(f"caught exception: {e}")

        # TODO: some of the data could already be cached, check other dates
        # find files in cache_dir that match the ticker_symbol
        for f in os.listdir(cache_dir):
            if f.startswith(f'{ticker_symbol}_') and f.endswith('.pickle'):
                print(f"Found file in cache: {f}") if verbose > 0 else None
                dataset = pd.read_pickle(f"{cache_dir}{f}")
                break
        
        if 'dataset' in locals():
            # check if the dataset is for the same date range
            if dataset.Date.min() <= start and dataset.Date.max() >= end:
                print(f"Returning cached data for {ticker_symbol
                        } from {dataset.Date.min()} to {dataset.Date.max()}") if verbose > 0 else None
                return dataset
            
            # check if the dataset has data for the start date
            if dataset.Date.min() <= start:
                print(f"Found cached data for {ticker_symbol} from {
                    dataset.Date.min()} to {dataset.Date.max()}") if verbose > 0 else None
                print(f"Getting data for {ticker_symbol} from {
                    dataset.Date.max()} to {end}") if verbose > 0 else None
                dataset = dataset.append(get_ticker_data(
                    ticker_symbol, start=dataset.Date.max(),
                    end=end, cache_dir=None))
                return dataset
            
            # check if the dataset has data for the end date
            if dataset.Date.max() >= end:
                print(f"Found cached data for {ticker_symbol} from {
                    dataset.Date.min()} to {dataset.Date.max()}") if verbose > 0 else None
                print(f"Getting data for {ticker_symbol} from {
                    start} to {dataset.Date.min()}") if verbose > 0 else None
                dataset = get_ticker_data(
                    ticker_symbol, start=start,
                    end=dataset.Date.min(), cache_dir=None).append(dataset)
                return dataset

    print(f"Getting ticker data for {ticker_symbol}")
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
    analyst_price_target = None
    earnings_forecast = None
    earnings_trend = None
    insider_purchases = None
    insider_roster_holders = None
    insider_transactions = None
    major_holders = None
    rev_forecast = None
    shares = None
    trend_details = None
    
    try:
        analyst_price_target = symbol.get_analyst_price_target(as_dict=True)
        print(f"analyst price target: {analyst_price_target}") if verbose > 1 else None
    except Exception as e:
        pass
    try:
        earnings_forecast = symbol.get_earnings_forecast(as_dict=True)
        print(f"earnings forecast: {earnings_forecast}") if verbose > 1 else None
    except Exception as e:
        pass
    try:
        earnings_trend = symbol.get_earnings_trend(as_dict=True)
        print(f"earnings trend: {earnings_trend}") if verbose > 1 else None
    except Exception as e:
        pass
    try:
        insider_purchases = symbol.get_insider_purchases(as_dict=True)
        print(f"insider purchases: {insider_purchases}") if verbose > 1 else None
    except Exception as e:
        pass
    try:
        insider_roster_holders = symbol.get_insider_roster_holders(as_dict=True)
        print(f"insider roster holders: {insider_roster_holders}") if verbose > 1 else None
    except Exception as e:
        pass
    try:
        insider_transactions = symbol.get_insider_transactions(as_dict=True)
        print(f"insider transactions: {insider_transactions}") if verbose > 1 else None
    except Exception as e:
        pass
    try:
        major_holders = symbol.get_major_holders(as_dict=True)
        print(f"major holders: {major_holders}") if verbose > 1 else None
    except Exception as e:
        pass
    try:
        rev_forecast = symbol.get_rev_forecast(as_dict=True)
        print(f"rev forecast: {rev_forecast}") if verbose > 1 else None
    except Exception as e:
        pass
    try:
        shares = symbol.get_shares(as_dict=True)
        print(f"shares: {shares}") if verbose > 1 else None
    except Exception as e:
        pass
    try:
        trend_details = symbol.get_trend_details(as_dict=True)
        print(f"trend details: {trend_details}") if verbose > 1 else None
    except Exception as e:
        pass

    result = np.array({
        'symbol': ticker_symbol,
        'data': {
            'actions': symbol.get_actions(),
            'analyst_price_target': analyst_price_target,
            'balance_sheet': symbol.get_balance_sheet(as_dict=True),
            'calendar': symbol.get_calendar(),
            'capital_gains': symbol.get_capital_gains(),
            'cash_flow': symbol.get_cash_flow(as_dict=True),
            'dividends': symbol.get_dividends(),
            'earnings_dates': symbol.get_earnings_dates(),
            'earnings_forecast': earnings_forecast,
            'earnings_trend': earnings_trend,
            'financials': symbol.get_financials(as_dict=True),
            'income_stmt': symbol.get_income_stmt(as_dict=True),
            'insider_purchases': insider_purchases,
            'insider_roster_holders': insider_roster_holders,
            'insider_transactions': insider_transactions,
            'institutional_holders': symbol.get_institutional_holders(as_dict=True),
            'major_holders': major_holders,
            'news': symbol.get_news(),
            'recommendations': symbol.get_recommendations(as_dict=True),
            'recommendations_summary': symbol.get_recommendations_summary(as_dict=True),
            'rev_forecast': rev_forecast,
            'shares': shares,
            'shares_full': symbol.get_shares_full(start=start, end=end),
            'splits': symbol.get_splits(),
            'trend_details': trend_details,
            'upgrades_downgrades': symbol.get_upgrades_downgrades(as_dict=True),
        },
        'history': hist
    })

    if isinstance(cache_dir, str):
        import pickle
        with open(cache_filename, 'wb') as f:
            pickle.dump(result, f)

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
def get_sp500_symbols():
    # get the S&P 500 symbols from wikipedia
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    sp500 = tables[0]
    symbols = sp500.Symbol.to_list()
    print(f"Got {len(symbols)} symbols: {symbols[:10]}")

def get_50_stock_symbols():
    # get 50 stock symbols
    syms = ['AAPL', 'AMZN', 'MSFT', 'MSTR', 'TSLA',
            'GOOGL', 'GOOG', 'META', 'NFLX', 'NVDA',
            'PYPL', 'ADBE', 'INTC', 'CSCO', 'CMCSA',
            'PEP', 'COST', 'AMGN', 'AVGO', 'TXN',
            'QCOM', 'GILD', 'SBUX', 'BKNG', 'MDLZ',
            'FISV', 'INTU', 'ADP', 'ISRG', 'REGN',
            'TMUS', 'AMD', 'MDB', 'CSX', 'ADI',
            'ILMN', 'BITX', 'BIIB', 'MU', 'BITI', 'CITI',
            'SPY', 'X', 'BSX', 'AA', 'A',
            'AAP', 'AAL', 'IWM', 'ABBV', 'ABT',
            ]
    return syms

def get_top_symbols_in_todays_news():
    # get the top stock symbols in today's news
    # url = "https://www.marketwatch.com/latest-news?mod=top_nav" # forbidden
    # url = 'https://google.com/finance' # no tables found
    # url = 'https://finance.yahoo.com/news/' # no tables found
    # url = 'https://www.benzinga.com/news/' # no tables found
    # url = 'https://www.bloomberg.com/markets' # no tables found
    # url = 'https://www.cnbc.com/finance/' # no tables found
    # url = 'https://stockanalysis.com/stocks/' # forbidden
    url = 'data/stockanalysis.html'
    tables = pd.read_html(url)
    news = tables[0]
    symbols = news.Symbol.to_list()
    print(f"Got {len(symbols)} symbols: {symbols[:10]+symbols[-10:]}")
    return symbols

def list_markets(verbose=False):
    # stock_symbols = get_50_stock_symbols()
    stock_symbols = get_top_symbols_in_todays_news()
    data = []
    for s in stock_symbols:
        try:
            data.append(get_ticker_data(s, verbose=verbose))
        except Exception as e:
            print(f"caught exception: {e}")
    return data


class YahooFinanceDataTest(tf.test.TestCase):
    # def test_yf(self):
    #     data = yahoofinance_data('AAPL', verbose=True)
    #     print(f"yahoofinance data: {data}")
    
        
    # def test_yf_multi(self):
    #     data = get_tickers(['AAPL', 'AMZN', 'MSFT', 'MSTR'])
    #     print(f"ticker data: {len(data)}")
    #     # summarize(data)

    def test_get_markets(self):
        # get all the different kinds of markets        
        markets = list_markets(verbose=1)
        print(f"len(markets): {len(markets)}")
        # print(f"markets: {markets.shape}")

    # def test_get_todays_newsworthy_symbols(self):
    #     symbols = get_top_symbols_in_todays_news()
    #     print(f"top symbols: {symbols[10:20]}")

if __name__ == "__main__":
    tf.test.main()
