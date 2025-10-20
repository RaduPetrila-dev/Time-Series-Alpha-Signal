
import yfinance as yf
import pandas as pd
from pathlib import Path

Path("data").mkdir(exist_ok=True, parents=True)

faang = ["AAPL","AMZN","GOOGL","META","NFLX"]
start = "2018-01-01"
end   = "2023-12-31"

px = yf.download(faang, start=start, end=end, interval="1d")["Adj Close"]
px = px.dropna(how="all")
px.to_csv("data/prices_faang_2018_2023.csv", index=True)
print("Saved data/prices_faang_2018_2023.csv with shape:", px.shape)
