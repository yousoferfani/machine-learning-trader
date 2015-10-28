import pandas as pd
import datetime

import google_quote
import neural_net_trader

data = google_quote.GoogleIntradayQuote('tsla', 300, 30)
data.write_csv("data.csv")
data = pd.read_csv("data.csv", names = ["SYMBOL", "DATE", "TIME", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"])

net = neural_net_trader.NeuralNet(10, 9, 1)
