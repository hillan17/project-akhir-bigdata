import os
import zipfile
import subprocess
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

os.makedirs("temp_raw/binance", exist_ok=True)
os.makedirs("temp_raw/yfinance", exist_ok=True)

def upload_to_hdfs(local_path, hdfs_dir):
    cmd = f"hdfs dfs -put -f {local_path} {hdfs_dir}/"
    subprocess.run(cmd, shell=True, check=True)
    print(f"[HDFS UPLOAD] {local_path} -> {hdfs_dir}/")

# ==============================================================================
# 1. AKUISISI BINANCE PUBLIC DATA (1 Tahun Terakhir: Nov 2025 - Nov 2026)
# ==============================================================================
symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
cols = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "num_trades",
    "taker_buy_base", "taker_buy_quote", "ignore"
]

monthly_periods = [
    "2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04",
    "2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10"
]
daily_dates = pd.date_range(start="2026-11-01", end=datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d").tolist()

for sym in symbols:
    df_list = []
    # Tarik arsip bulanan
    for m in monthly_periods:
        url = f"https://data.binance.vision/data/spot/monthly/klines/{sym}/1h/{sym}-1h-{m}.zip"
        r = requests.get(url)
        if r.status_code == 200:
            zpath = f"temp_raw/binance/{sym}-{m}.zip"
            with open(zpath, "wb") as f:
                f.write(r.content)
            with zipfile.ZipFile(zpath, "r") as z:
                df_list.append(pd.read_csv(z.open(z.namelist()[0]), names=cols))
            os.remove(zpath)

    # Tarik arsip harian bulan berjalan (Nov 2026)
    for d in daily_dates:
        url = f"https://data.binance.vision/data/spot/daily/klines/{sym}/1h/{sym}-1h-{d}.zip"
        r = requests.get(url)
        if r.status_code == 200:
            zpath = f"temp_raw/binance/{sym}-{d}.zip"
            with open(zpath, "wb") as f:
                f.write(r.content)
            with zipfile.ZipFile(zpath, "r") as z:
                df_list.append(pd.read_csv(z.open(z.namelist()[0]), names=cols))
            os.remove(zpath)

    if df_list:
        full_df = pd.concat(df_list, ignore_index=True)
        unit = "us" if full_df["open_time"].iloc[0] > 1e14 else "ms"
        full_df["timestamp"] = pd.to_datetime(full_df["open_time"], unit=unit, utc=True)
        full_df["symbol"] = sym
        
        local_csv = f"temp_raw/binance/{sym}_1y_1h.csv"
        full_df.to_csv(local_csv, index=False)
        upload_to_hdfs(local_csv, "/datalake/raw/binance")

# ==============================================================================
# 2. AKUISISI YAHOO FINANCE API (Data Makro & Saham 1 Tahun Terakhir)
# ==============================================================================
tickers = {
    "SP500": "^GSPC",
    "DXY": "DX-Y.NYB",
    "GOLD": "GC=F",
    "IHSG": "^JKSE"
}

end_date = datetime.now()
start_date = end_date - timedelta(days=365)

for name, ticker in tickers.items():
    df_yf = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        interval="1h",
        progress=False
    )
    df_yf.reset_index(inplace=True)
    df_yf.columns = [c[0] if isinstance(c, tuple) else c for c in df_yf.columns]
    
    local_csv = f"temp_raw/yfinance/{name}_1y_1h.csv"
    df_yf.to_csv(local_csv, index=False)
    upload_to_hdfs(local_csv, "/datalake/raw/yfinance")

print("Seluruh dataset 1 tahun berhasil diakuisisi ke Hadoop HDFS!")