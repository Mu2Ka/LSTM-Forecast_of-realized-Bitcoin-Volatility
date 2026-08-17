import io
import pandas as pd
import zipfile
import requests

symbol = "BTCUSDT"
interval = "1h"
years = range(2020,2026)
all_data = []
columns = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]
monthly_data = []
for year in years:
    for month in range(1, 13):
        filename= f"{symbol}-{interval}-{year}-{month:02d}.zip"
        url = (
            "https://data.binance.vision/data/spot/monthly/klines/"
            f"{symbol}/{interval}/{filename}"
        )
        response = requests.get(url,timeout=50)
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_name = z.namelist()[0]
            with z.open(csv_name) as csv_file:
                month_df = pd.read_csv(
                    csv_file,
                    header=None,
                    names=columns,
                )
        timestamp_unit = 'us' if year >= 2025 else 'ms'
        month_df['timestamp'] = pd.to_datetime(month_df['open_time'], unit=timestamp_unit,utc=True)
        monthly_data.append(month_df)
df = pd.concat(monthly_data, ignore_index=True)
df.to_csv(
    "BTCUSDT_1h_2020_2025.csv",
    index=False,
)