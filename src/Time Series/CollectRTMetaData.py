import requests
import pandas as pd
import time
import numpy as np

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
    # start_time = end_time - step_count * 60 * 1000  # 3 years back
    interval_co = 60
    if interval is '1h':
        interval_co=60*60
    elif interval is '1d':
        interval_co=24*60*60
    
    start_time = end_time - step_count * interval_co * 1000  # 3 years back
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

def get_coinbase_historical_klines(product_id, start, end, granularity=60):
    """
    Downloads historical data for a given symbol from Coinbase.
    """
    url = f'https://api.exchange.coinbase.com/products/{product_id}/candles'
    params = {
        'start': start,
        'end': end,
        'granularity': granularity  # 60 seconds = 1 minute
    }
    response = requests.get(url, params=params,timeout=30)
    return response.json()

##Endtime should be in miliseconds
def fetch_coinbase_data(product_id,data_count,end_time=None):
    """
    Fetch 3 years of 1-minute interval data.
    """

    if(end_time==None):
        end_time = pd.Timestamp.now(tz='UTC')
    else:
        end_time = pd.to_datetime(end_time, unit='ms', utc=True)
        
    start_time = end_time - pd.Timedelta(minutes=data_count)
    # start_time = end_time - pd.Timedelta(days=1)
    # start_time = end_time - pd.Timedelta(minutes=600)
    counter = 0
    all_data = []
    while start_time < end_time:
        # next_start_time = start_time + pd.Timedelta(minutes=data_count)  # Each request only gives 300 data points
        next_start_time = min(start_time + pd.Timedelta(minutes=300), end_time)
        print(start_time)
        print(next_start_time)

        print("A1")
        candles = get_coinbase_historical_klines(product_id, start_time.isoformat(), next_start_time.isoformat())
        print(candles)
        print("A2")
        if not candles:
            print("Candles null")
            break
        all_data.extend(candles)
        start_time = next_start_time

        counter+=1
        if(counter%10==0):
            print(counter,product_id)
        if(counter%200==0):
            time.sleep(60)

        time.sleep(1)  # To avoid rate limiting
        # time.sleep(0.3)  # To avoid rate limiting

       

        if next_start_time>=end_time:
            break

    # print(all_data)
    
    df = pd.DataFrame(all_data, columns=['Time', 'Low', 'High', 'Open', 'Close', 'Volume'])
    df = df.sort_values('Time')
    df = df.reset_index()
    print(df.count)
    # df = pd.DataFrame(all_data)
    return df

timenow = int(time.time() * 1000)

# binance_file_list = [
#     "binance_btcusdt_3_years.csv",
#     "binance_ethusdt_3_years.csv",
#     "binance_solusdt_3_years.csv",
#     "binance_avaxusdt_3_years.csv",
#     "binance_dotusdt_3_years.csv",
#     "binance_adausdt_3_years.csv",
#     "binance_bnbusdt_3_years.csv",
#     "binance_linkusdt_3_years.csv",
#     "binance_xrpusdt_3_years.csv",
#     "binance_etcusdt_3_years.csv",
#     "binance_gttusdt_3_years.csv",
#     "binance_ltcusdt_3_years.csv",
#     "binance_nearusdt_3_years.csv",
#     "binance_thetausdt_3_years.csv",
#     "binance_trxusdt_3_years.csv",
    
# ]

def fill_missing_timestamps(
    df,
    time_column='timestamp',
    freq='1T',
    fill_method='ffill',
    unit='ms',
    tz='UTC'
):
    """
    Fills missing timestamps in a candle DataFrame by reindexing and filling missing data.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing candle data with a timestamp column.
        
    time_column : str, optional (default='timestamp')
        The name of the column containing timestamp data.
        
    freq : str, optional (default='1T')
        The frequency for reindexing. '1T' stands for 1 minute.
        Examples:
            - '1T' : 1 minute
            - '5T' : 5 minutes
            - '1H' : 1 hour
            - '15S': 15 seconds
            - '500ms': 500 milliseconds
        
    fill_method : str or None, optional (default='ffill')
        The method used to fill missing data after reindexing.
        Options include:
            - 'ffill' : Forward fill
            - 'bfill' : Backward fill
            - None    : Do not fill missing data (leave NaNs)
        
    unit : str, optional (default='ms')
        The unit of the timestamp data. Common options:
            - 'ms' : milliseconds
            - 's'  : seconds
            - 'ns' : nanoseconds
        
    tz : str or None, optional (default='UTC')
        Timezone to localize the timestamps. Use 'UTC' for Coordinated Universal Time.
        If your data is in a different timezone, specify it here (e.g., 'Europe/Istanbul').
        Use None if the timestamps are timezone-naive.

    Returns:
    --------
    pd.DataFrame
        A new DataFrame with missing timestamps filled and data populated accordingly.
    """
    
    # Step 1: Convert the timestamp column to datetime, specifying the unit and timezone
    df[time_column] = pd.to_datetime(df[time_column], unit=unit, utc=True)
    
    # Step 2: Drop duplicate timestamps to avoid issues during reindexing
    df = df.drop_duplicates(subset=[time_column])
    
    # Step 3: Set the timestamp column as the index
    df = df.set_index(time_column)
    
    # Step 4: Sort the DataFrame by the index to ensure chronological order
    df = df.sort_index()
    
    # Step 5: Create a complete range of timestamps at the specified frequency
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq, tz=tz)
    
    # Step 6: Reindex the DataFrame to include all timestamps in the full range
    df = df.reindex(full_range)
    
    # Step 7: Fill missing data using the specified fill method
    if fill_method:
        df.fillna(method=fill_method, inplace=True)
    
    # Step 8: Reset the index to convert the timestamp back to a column
    df = df.reset_index().rename(columns={'index': time_column})

    print(df.count)
    
    return df

# binance_symbols = ['BTCUSDT']
binance_symbols = ['BTCUSDT','ETHUSDT','SOLUSDT','AVAXUSDT','DOTUSDT','ADAUSDT','BNBUSDT','LINKUSDT','XRPUSDT','ETCUSDT','GRTUSDT','LTCUSDT','NEARUSDT','THETAUSDT','TRXUSDT']

# coinbase_symbols = ['BTCUSDT']
coinbase_symbols = ['BTC-USD']

df = None

def fetch_binance_datas_extended(symbols,freq='1m',data_count=40):
    for symbol in symbols:
        dfc = fetch_binance_data(symbol, freq,data_count,end_time=timenow)
        print(dfc.head())
        # if df is None:
        #     time_col = dfc['Time']
        time_col = dfc['Time']
        # prefix = 'binance_' + symbol.lower()+'_'+freq+'_'
        prefix = 'binance_' + symbol+'_'
        dfc[prefix+'Open']=dfc['Open']
        dfc[prefix+'High']=dfc['High']
        dfc[prefix+'Low']=dfc['Low']
        dfc[prefix+'Close']=dfc['Close']
        dfc[prefix+'Volume']=dfc['Volume']
        dfc = dfc[[prefix+'Open',prefix+'High',prefix+'Low',prefix+'Close',prefix+'Volume']]
        dfc.insert(loc=0, column='Time', value=time_col)
        return dfc
        if df is None:
            df = dfc
            df.insert(loc=0, column='Time', value=time_col)
            continue
        df = pd.concat([df,dfc],axis=1)
    print(df.count)
    return df
def fetch_binance_datas(symbols,df):
    for symbol in symbols:
        dfc = fetch_binance_data(symbol, '1m',40,end_time=timenow)
        # print(dfc.head())
        print(f"Fetched {symbol} data from binance")
        if df is None:
            time_col = dfc['Time']
        prefix = 'binance_' + symbol.lower()+'_'
        dfc[prefix+'Open']=dfc['Open']
        dfc[prefix+'High']=dfc['High']
        dfc[prefix+'Low']=dfc['Low']
        dfc[prefix+'Close']=dfc['Close']
        dfc[prefix+'Volume']=dfc['Volume']
        dfc = dfc[[prefix+'Open',prefix+'High',prefix+'Low',prefix+'Close',prefix+'Volume']]
        if df is None:
            df = dfc
            df.insert(loc=0, column='Time', value=time_col)
            print("Created df")
            continue
        df = pd.concat([df,dfc],axis=1)
        print("Merged with df")
    # print(df.count)
    print("Fetched binance datas")
    return df

def add_coinbase_data(symbols : list[str],df):
    for symbol in symbols:
        dfc = fetch_coinbase_data(symbol,40,end_time=timenow)
        # print(dfc.head())
        print(f"Fetched {symbol} data from binance")

        if df is None:
            time_col = dfc['Time']
        symbol2 = symbol.replace('-','')
        prefix = 'coinbase_' + symbol2.lower()+'_'
        dfc[prefix+'Open']=dfc['Open']
        dfc[prefix+'High']=dfc['High']
        dfc[prefix+'Low']=dfc['Low']
        dfc[prefix+'Close']=dfc['Close']
        dfc[prefix+'Volume']=dfc['Volume']
        dfc = dfc[[prefix+'Open',prefix+'High',prefix+'Low',prefix+'Close',prefix+'Volume']]
        if df is None:
            df = dfc
            df.insert(loc=0, column='Time', value=time_col)
            print("Created df")
            continue
        df = pd.concat([df,dfc],axis=1)
        print("Merged with df")
    # print(df.count)
    print("Fetched coinbase data")
    return df

def add_daytime_encoding(df : pd.DataFrame):
    df["datetime"] = pd.to_datetime( df["Time"],unit='s')
    hour = df["datetime"].dt.hour
    minute = df["datetime"].dt.minute + hour*60

    # Normalize the minutes to the range [0, 2π]
    normalized_minutes = (minute / (24 * 60)) * 2 * np.pi
        
    # Add sine and cosine of the normalized time as new columns
    df['minute_cos'] = np.cos(normalized_minutes)
    df['minute_sin'] = np.sin(normalized_minutes)

    df.drop("datetime",axis=1)

def RSI_From_Data(df : pd.DataFrame,window_size=14):
        # Step 3: Separate positive and negative changes
        diffs = df.diff()
        gains = diffs.clip(lower=0)  # Positive changes
        losses = -diffs.clip(upper=0)  # Negative changes (inverted sign)

        avg_gain = gains.rolling(window=window_size, min_periods=window_size).mean()
        avg_loss = losses.rolling(window=window_size, min_periods=window_size).mean()

        #Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        print("RSI coming")
        print(diffs.to_string())
        print(rsi.to_string())
        print("RSI came")

        return rsi

rsi_metas = ["btcusdt"]
        
def _calculate_RSI(df,metas2calculate):
    # df.set_index('datetime', inplace=True)

    hourly_resampled = fetch_binance_datas_extended(metas2calculate,'1h',15)
    daily_resampled = fetch_binance_datas_extended(metas2calculate,'1d',15)
    
    print("hourly data")
    print(hourly_resampled.to_string())
    print("close part")
    print(hourly_resampled[f'binance_{metas2calculate[0]}_Close'].to_string())

    # hourly_resampled = df.resample('H').last()
    # daily_resampled = df.resample('D').last()
    for meta in metas2calculate:
        meta = 'binance_'+meta
        hourly_resampled[f'{meta}_Close'] = pd.to_numeric(hourly_resampled[f'{meta}_Close'], errors='coerce')
        daily_resampled[f'{meta}_Close'] = pd.to_numeric(daily_resampled[f'{meta}_Close'], errors='coerce')
        df[f'{meta}_RSI_HOURLY'] = RSI_From_Data(hourly_resampled[f'{meta}_Close'])
        df[f'{meta}_RSI_DAILY'] = RSI_From_Data(daily_resampled[f'{meta}_Close'])
        
        df[f'{meta}_RSI_HOURLY'] = df[f'{meta}_RSI_HOURLY'].fillna(method='bfill')
        df[f'{meta}_RSI_HOURLY'] = df[f'{meta}_RSI_HOURLY'].fillna(method='ffill')
        df[f'{meta}_RSI_DAILY'] = df[f'{meta}_RSI_DAILY'].fillna(method='bfill')
        df[f'{meta}_RSI_DAILY'] = df[f'{meta}_RSI_DAILY'].fillna(method='ffill')
    # df.to_csv("after_RSI.csv")
    # valid_indices = df[f'{metas2calculate[0]}_RSI_DAILY'].dropna().index
    print("Count before")
    # self.data = self.data.loc[valid_indices]
    print("Count after")
    # print(self.data.count)

def PrepeareRTData():
    df=None
    df = fetch_binance_datas(binance_symbols,df)
    df = add_coinbase_data(coinbase_symbols,df)
    df = fill_missing_timestamps(df,time_column='Time',freq='1T')
    add_daytime_encoding(df)
    _calculate_RSI(df,rsi_metas)
    return df



if __name__ == "__main__":
    print("About do create current 40 data point with extra features")
    PrepeareRTData()
    df.to_csv("rt_crtypto_test.csv")



print("End")

    



