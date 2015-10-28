import datetime
import pandas as pd

import google_quote
import neural_net
import neural_net_trainer
import data_entry

data = google_quote.GoogleIntradayQuote('tsla', 300, 30)
data.write_csv("data.csv")
data = pd.read_csv("data.csv", names = ["SYMBOL", "DATE", "TIME", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"])

# Creates a neural net with i input nodes, j hidden nodes, and k output nodes
i = 10
j = 10
k = 1
net = neural_net.NeuralNet(i, j, k)
trainer = neural_net_trainer.NeuralNetTrainer(net)

# We now need to prep the data into training sets and target sets
# Training sets will consist of a set of ten stock values and an indicator as to whether holding 
# the stock from time 10 to 11 will result in a gain or a loss; super simplistic classificaiton
