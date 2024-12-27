import requests
import pandas as pd
import time

def get_binance_symbols():
    url = 'https://api.binance.com/api/v3/exchangeInfo'
    response = requests.get(url)
    data = response.json()
    symbols = [symbol['symbol'] for symbol in data['symbols']]
    return symbols

# binance_symbols = get_binance_symbols()
# print(f"Total Binance Symbols: {len(binance_symbols)}")
# print(binance_symbols[:50])  # Display first 10 symbols

def get_binance_historical_klines(symbol, interval, start_time, end_time=None):
    """
    Downloads historical data for a given symbol from Binance.
    """
    url = 'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': symbol.upper(),
        'interval': interval,
        'startTime': start_time,
        'endTime': end_time,
        'limit': 1000  # Max 1000 per request
    }
    response = requests.get(url, params=params)
    return response.json()

def fetch_binance_data(symbol, interval,step_count,end_time=None):
    """
    Fetch 3 years of 1-minute interval data.
    """

    counter=0
    if end_time == None:
        end_time = int(time.time() * 1000)  # Current time in milliseconds
    start_time = end_time - step_count * 60 * 1000  # 3 years back
    # start_time = end_time - 3 * 365 * 1000
    all_data = []
    while start_time < end_time:
        klines = get_binance_historical_klines(symbol, interval, start_time, end_time)
        if not klines:
            break
        all_data.extend(klines)
        start_time = int(klines[-1][0]) + 1  # Move to the next candle
        counter+=1
        if counter % 100 == 0:
            print(symbol)
            print("Counter:",counter)

        time.sleep(0.2)  # To avoid rate limiting
    df = pd.DataFrame(all_data, columns=['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
    return df[['Time','Open','High','Low','Close','Volume']]

timenow = int(time.time() * 1000)

binance_symbols = ['BTCUSDT','SOLUSDT','ETHUSDT','XRPUSDT','DOTUSDT','AVAXUSDT','THETAUSDT','BNBUSDT','LINKUSDT','ADAUSDT','ETCUSDT','NEARUSDT','LTCUSDT','TRXUSDT','GRTUSDT']
df = None
def fetch_binance_datas(symbols,df):
    for symbol in symbols:
        dfc = fetch_binance_data(symbol, '1m',40,end_time=timenow)
        print(dfc.head())
        if df is None:
            df = dfc
            continue
        df = pd.concat([df,dfc],axis=1)
    print(df.count)
    return df
df = fetch_binance_datas(binance_symbols,df)

print("End")

    



