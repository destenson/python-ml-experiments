import tensorflow as tf
import yfinance as yf
import numpy as np
import pandas as pd

import datetime as dt
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
def yahoofinance_data(symbol, start=None, end=None, verbose=False):
    dataset = get_ticker_data(symbol, start=start, end=end, verbose=verbose)
    
    print(f"dataset: {dataset}") if verbose > 1 else None
    
    return dataset

def get_tickers(symbols, start=None, end=None, verbose=False):
    datasets = []
    for symbol in symbols:
        dataset = get_ticker_data(symbol, start=start, end=end, verbose=verbose)
        datasets.append(dataset)
        print(f"dataset: {dataset}") if verbose > 0 else None
    return np.array(datasets)

def update_dataset(original, new, verbose=1):
    if original is None:
        print("original is None") if verbose > 1 else None
        return new
    if new is None:
        print("new is None") if verbose > 1 else None
        return original
    if isinstance(original, dict) and isinstance(new, dict):
        print("Both are dict types") if verbose > 0 else None
        if new.keys() == original.keys():
            print("Same keys") if verbose > 0 else None
            for k in new.keys():
                if type(original[k]) == type(new[k]):
                    original[k] = update_dataset(original[k], new[k], verbose=verbose)
                else:
                    print(f"Different types for {k}: {type(original[k])} and {type(new[k])}") if verbose > 1 else None
                    raise ValueError(f"Different types for {k}: {type(original[k])} and {type(new[k])}")
            return original
        else:
            print("Different keys") if verbose > 0 else None
            print(f"New keys: {new.keys()}") if verbose > 0 else None
            print(f"Original keys: {original.keys()}") if verbose > 0 else None
            for k in new.keys():
                if k in original.keys() and not isinstance(original[k], str):
                    if type(original[k]) == type(new[k]):
                        original[k] = update_dataset(original[k], new[k], verbose=verbose)
                    else:
                        print(f"Different types for {k}: {type(original[k])} and {type(new[k])}") if verbose > 1 else None
                        raise ValueError(f"Different types for {k}: {type(original[k])} and {type(new[k])}")
                else:
                    if type(original[k]) == type(new[k]):
                        print(f"Updating {k} from {original[k]} to {new[k]}") if verbose > 1 else None
                        original[k] = new[k]
                    else:
                        raise ValueError(f"Invalid types for {k}: {type(original[k])} and {type(new[k])}")
                        # print(f"Invalid types: {type(original[k])} and {type(new[k])}") if verbose > 1 else None
            return original
    if isinstance(original, pd.DataFrame) and isinstance(new, pd.DataFrame):
        print("Both are DataFrame types") if verbose > 1 else None
        print(f"{original.head()} .. {original.tail()}") if verbose > 1 else None
        print(f"{new.head()} .. {new.tail()}") if verbose > 1 else None
        if len(original) == 0:
            print("Empty original") if verbose > 0 else None
            return new
        if len(new) == 0:
            print("Empty new") if verbose > 0 else None
            return original
        print("concat'ing DataFrames") if verbose > 1 else None
        return pd.concat([original, new])
    if isinstance(original, list) and isinstance(new, list):
        print("Both are list types") if verbose > 0 else None
        
        return original + new
    if isinstance(original, np.ndarray) and isinstance(new, np.ndarray):
        print("Both are ndarray types") if verbose > 0 else None
        return np.concatenate([original, new])
    if type(original) == type(new):
        print(f"Both are same type: {type(original)}") if verbose > 1 else None
        return new
    raise ValueError(f"Invalid types: {type(original)} and {type(new)}")
    # print("Returning new") if verbose > 0 else None
    # return new

def get_last_business_day(date):
    if isinstance(date, str):
        date = pd.to_datetime(date)
    if date.weekday() > 4:
        # Create an offset of 5 Business days
        bd = pd.tseries.offsets.BusinessDay(n = 5)
        date = bd.rollback(date)

    return date.strftime('%Y-%m-%d')        

def get_ipo_date(ticker_symbol, dataset=None):
    if dataset is not None:
        if isinstance(dataset, dict):
            if 'info' in dataset.keys():
                info = dataset['info']
                ipo_date = info['firstTradeDateEpochUtc']
            else:
                info = Ticker(ticker_symbol, session=session).info
                ipo_date = info['firstTradeDateEpochUtc']
                dataset = update_dataset(dataset, {'info': info})
                # raise ValueError(f"Invalid dataset, expected 'info' in keys: {dataset.keys()}")
        elif isinstance(dataset, np.ndarray):
            assert dataset.size == 1, f"Expected dataset to have size 1, got {dataset.size}"
            # print(f"dataset.item(1): {dataset.item(1)}") #if verbose > 0 else None
            if 'info' in dataset.item(0).keys():
                info = dataset.item(0)['info']
                ipo_date = info['firstTradeDateEpochUtc']
            else:
                info = Ticker(ticker_symbol, session=session).info
                try:
                    ipo_date = info['firstTradeDateEpochUtc']
                    dataset = update_dataset(dataset, {'info': info})
                except Exception as e:
                    print(f"caught exception: {e}")
                    ipo_date = -1
        else:
            raise ValueError(f"Invalid dataset type: {type(dataset)}")
    else:
        info = Ticker(ticker_symbol, session=session).info
        ipo_date = info['firstTradeDateEpochUtc']
    # print(f"{ticker_symbol} IPO date: ({ipo_date})") #if verbose > 0 else None
    if ipo_date < 0:
        ipo_date = dt.datetime.fromtimestamp(0, dt.UTC) - (dt.datetime.fromtimestamp(-ipo_date, dt.UTC) - dt.datetime.fromtimestamp(0, dt.UTC))
    else:
        ipo_date = dt.datetime.fromtimestamp(ipo_date, dt.UTC)
    # print(f"{ticker_symbol} IPO date: {ipo_date.date()} ({ipo_date})") #if verbose > 0 else None
    return ipo_date, info, dataset


def update_stock_info(data):
    if isinstance(data, dict):
        if 'info' not in data.keys():
            info = Ticker(data['symbol'], session=session).info
            info = update_dataset(info, {'retrieval_date': pd.to_datetime(date.today()).date()})
            data = update_dataset(data, {'info': info})
        else:
            if data['info']['retrieval_date'] < pd.to_datetime(date.today()).date()-pd.tseries.offsets.Day(7):
                info = Ticker(data['symbol'], session=session).info
                info = update_dataset(info, {'retrieval_date': pd.to_datetime(date.today()).date()})
                data = update_dataset(data, {'info': info})
            else:
                print("info already in data") #if verbose > 0 else None
    else:
        print(f"Invalid data type: {type(data)}")
    return data

def get_ticker_data(ticker_symbol,
                    start=None, end=date.today().strftime('%Y-%m-%d'),#-dt.timedelta(days=1),
                    period="max",
                    interval='1d',
                    cache_dir='data/',
                    verbose=1):
    dataset = None
    if start is None:
        if cache_dir is None:
            raise ValueError("start date is required when cache_dir is None")
        import os
        for f in os.listdir(cache_dir):
            if f.startswith(f'{ticker_symbol}_') and f.endswith('.pickle'):
                print(f"Found file in cache: {f}") if verbose > 0 else None
                loaded_filename = f'{cache_dir}{f}'
                dataset = pd.read_pickle(loaded_filename)
                break
        start, info, _ds = get_ipo_date(ticker_symbol, dataset)
    if end is None:
        end = date.today()

    assert interval == '1d', f"Only daily interval is currently supported, got '{interval}'"
    start = pd.to_datetime(start).date()
    end = pd.to_datetime(end).date()
    # end is a weekend day, move it to the previous Friday
    if end.weekday() > 4:
        # Create an offset of 5 Business days (this is kind of fucking dumb)
        bd = pd.tseries.offsets.BusinessDay(n = 5, normalize=True)
        print(f"end was {end}") if verbose > 1 else None
        if end == date.today():
            print("End date is today") if verbose > 1 else None
            end = (bd.rollback(end) - pd.tseries.offsets.Day(1)).date()
        else:
            end = (bd.rollback(end) - pd.tseries.offsets.Day(1)).date()
        print(f"end is now {end}") if verbose > 1 else None
        if start >= end:
            print(f"start {start} >= end {end}")
            return np.array({})
    if isinstance(cache_dir, str):
        cache_filename = f"{cache_dir}{ticker_symbol}_{start}_{end}.pickle"
        try:
            import os
            if cache_filename and os.path.isfile(cache_filename):
                print(f"Loading data from cache: {cache_filename}") if verbose > 0 else None
                dataset = pd.read_pickle(cache_filename)
                # TODO: ensure the dataset contains the requried dates
                return dataset
        except Exception as e:
            print(f"caught exception: {e}")

        # some of the data could already be cached, check other dates
        # by finding files in cache_dir that match the ticker_symbol
        loaded_filename = None
        for f in os.listdir(cache_dir):
            if f.startswith(f'{ticker_symbol}_') and f.endswith('.pickle'):
                print(f"Found file in cache: {f}") if verbose > 0 else None
                loaded_filename = f'{cache_dir}{f}'
                dataset = pd.read_pickle(loaded_filename)
                break

        if 'dataset' in locals() and dataset is not None:
            if not isinstance(dataset, dict) and dataset.shape == ():
                dataset = dataset[()]
            # print(f"dataset type: {type(dataset)}") #if verbose > 0 else None
            if not isinstance(dataset, dict):
                raise ValueError(f"Expected dataset to be dict, got {type(dataset)}")
            # else:
            #     print(f"dataset keys: {dataset.keys()}") if verbose > 1 else None

            if not (isinstance(dataset, dict) and 'symbol' in dataset.keys()):
                raise ValueError(f"Invalid dataset, expected dict, got {type(dataset)}")
            else:
                # print(f"Keys: {dataset.keys()}") #if verbose > 0 else None

                print(f"Checking cached data for {dataset['symbol']}") if verbose > 0 else None

                if not 'history' in dataset.keys():
                    raise ValueError(f"Invalid dataset, expected history in keys: {dataset.keys()}")
                else:
                    history = dataset['history']
                    if not isinstance(history, pd.DataFrame) and not isinstance(history, pd.Index):                        
                        raise ValueError(f"Invalid history, expected DataFrame, got {type(history)}")
                    else:
                        print(f"History.index {history.index}") if verbose > 1 else None
                        print(f"Columns: {history.columns}") if verbose > 1 else None
                        if isinstance(history.index, pd.Index) and history.index.name != 'Date':
                            raise ValueError(f"Invalid history, expected Date in columns: {history.columns}")                            
                        if not isinstance(history.index, pd.DatetimeIndex) and not isinstance(history.index, pd.Index):
                            print(f"History.index {history.index}") if verbose < 2 else None
                            print(f"Columns: {history.columns}") if verbose < 2 else None
                            raise ValueError(f"Invalid history, expected Date in columns: {history.columns}")
                        else:
                            start_orig = start
                            if start_orig == '2023-01-01':
                                print("Start date is 2023-01-01") #if verbose > 0 else None
                            if end == date.today():
                                print("End date is today") #if verbose > 0 else None
                            start = pd.to_datetime(start).tz_localize('EST', ambiguous='raise')
                            end = pd.to_datetime(end).tz_localize('EST', ambiguous='raise')
                            # Create an offset of 5 Business days (this is kind of fucking dumb, especially since it doesn't work right)
                            bd = pd.tseries.offsets.BusinessDay(n = 5)
                            start = bd.rollforward(start)
                            if end == date.today():
                                print("End date is today") #if verbose > 0 else None
                                end = bd.rollback(end) - pd.tseries.offsets.DateOffset(days=1)
                            if start_orig == '2023-01-01':
                                start = start + pd.tseries.offsets.Day(1)
                            if len(history) == 0:
                                start = start.strftime('%Y-%m-%d')
                                end = end.strftime('%Y-%m-%d')
                                print(f"Empty history for {ticker_symbol}") if verbose > 0 else None
                                return get_ticker_data(ticker_symbol, start=start, end=end, cache_dir=None)[()]
                            date_min = history.index.min()
                            date_max = history.index.max()
                            print(f"Date range: {date_min} to {date_max}") if verbose > 0 else None
                            print(f"Start: {start} to {end}") if verbose > 0 else None
                            print(f"{history.head()} .. {history.tail()}") if verbose > 1 else None

                            if date_min <= start and date_max >= end:
                                print(f"Returning cached data for {ticker_symbol
                                        } from {date_min} to {date_max}") if verbose > 0 else None
                                return dataset
                            
                            # check if the dataset has data for the start date
                            if date_min <= start:
                                end = end.strftime('%Y-%m-%d')
                                print(f"Found cached data for {ticker_symbol} from after start {
                                    date_min} to {date_max}") if verbose > 0 else None
                                print(f"Getting data for {ticker_symbol} from {
                                    date_max} to {end}") if verbose > 0 else None
                                dataset = update_dataset(
                                    dataset, get_ticker_data(ticker_symbol,
                                                             start=date_max,
                                                             end=end,
                                                             cache_dir=None)[()])
                                with open(cache_filename, 'wb') as f:
                                    pd.to_pickle(dataset, f)
                                if loaded_filename is not cache_filename:
                                    print(f"Removing previously loaded file: {loaded_filename}") if verbose > 0 else None
                                    os.remove(loaded_filename)
                                return dataset

                            # check if the dataset has data for the end date
                            if date_max >= end:
                                start = start.strftime('%Y-%m-%d')
                                print(f"Found cached data for {ticker_symbol} from after end {
                                    date_min} to {date_max}") if verbose > 0 else None
                                print(f"Getting data for {ticker_symbol} from {
                                    start} to {date_min}") if verbose > 0 else None
                                dataset = update_dataset(
                                    get_ticker_data(ticker_symbol,
                                        start=start, end=date_min,
                                        cache_dir=None)[()], dataset)
                                with open(cache_filename, 'wb') as f:
                                    pd.to_pickle(dataset, f)
                                if loaded_filename is not cache_filename:
                                    print(f"Removing previously loaded file: {loaded_filename}") if verbose > 0 else None
                                    os.remove(loaded_filename)
                                return dataset
                            
                            if start < date_min and end > date_max:
                                start = start.strftime('%Y-%m-%d')
                                end = end.strftime('%Y-%m-%d')
                                print(f"Found cached data for {ticker_symbol} from after start to before end {
                                    date_min} to {date_max}") if verbose > 0 else None
                                # print(f"Getting data for {ticker_symbol} from {
                                #     date_min} to {date_max}") if verbose > 0 else None
                                dataset = update_dataset(
                                    get_ticker_data(ticker_symbol, start=start, end=date_min, cache_dir=None)[()], dataset)
                                dataset = update_dataset(dataset, get_ticker_data(
                                    ticker_symbol, start=date_max, end=end, cache_dir=None)[()])
                                with open(cache_filename, 'wb') as f:
                                    pd.to_pickle(dataset, f)
                                if loaded_filename is not cache_filename:
                                    print(f"Removing previously loaded file: {loaded_filename}") if verbose > 0 else None
                                    os.remove(loaded_filename)
                                return dataset

    if 'dataset' in locals() and dataset is not None:
        raise ValueError(f"We should've returned by now")

    print(f"Getting ticker data for {ticker_symbol} (cache_dir={cache_dir})") if verbose > 0 else None
    # create Ticker object    
    # symbol = Ticker(ticker_symbol)
    symbol = Ticker(ticker_symbol, session=session)

    # # get all stock info
    # symbol.info
    
    # make sure start and end are str
    start = start.strftime('%Y-%m-%d') if isinstance(start, dt.datetime) else start
    end = end.strftime('%Y-%m-%d') if isinstance(end, dt.datetime) else end

    print(f"Getting history from {start} to {end}") if verbose > 0 else None
    if start == end:
        print(f"Start date {start} is the same as end date {end}") if verbose > 0 else None
        return np.array({})
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
    institutional_holders = None
    major_holders = None
    rev_forecast = None
    shares = None
    shares_full = None
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
        institutional_holders = symbol.get_institutional_holders(as_dict=True)
        print(f"institutional holders: {institutional_holders}") if verbose > 1 else None
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
        shares_full = symbol.get_shares_full(start=start, end=end)
        print(f"shares full: {shares_full}") if verbose > 1 else None
    except Exception as e:
        pass
    try:
        trend_details = symbol.get_trend_details(as_dict=True)
        print(f"trend details: {trend_details}") if verbose > 1 else None
    except Exception as e:
        pass

    result = np.array({
        'symbol': ticker_symbol,
        'info': symbol.info,
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
            'institutional_holders': institutional_holders,
            'major_holders': major_holders,
            'news': symbol.get_news(),
            'recommendations': symbol.get_recommendations(as_dict=True),
            'recommendations_summary': symbol.get_recommendations_summary(as_dict=True),
            'rev_forecast': rev_forecast,
            'shares': shares,
            'shares_full': shares_full,
            'splits': symbol.get_splits(),
            'trend_details': trend_details,
            'upgrades_downgrades': symbol.get_upgrades_downgrades(as_dict=True),
        },
        'history': hist
    })

    if isinstance(cache_dir, str):
        import pickle
        with open(cache_filename, 'wb') as f:
            pd.to_pickle(result, f)
        if loaded_filename is not None:
            print(f"Removing previously loaded file: {loaded_filename}") if verbose > 0 else None
            os.remove(loaded_filename)

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
        # try:
        data.append(yahoofinance_data(s, verbose=verbose))
        # except Exception as e:
        #     print(f"caught exception: {e}")
    return data


class YahooFinanceDataTest(tf.test.TestCase):
    # def test_yf(self):
    #     data = yahoofinance_data('ABCL', verbose=True)
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
