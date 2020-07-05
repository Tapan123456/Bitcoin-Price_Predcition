import pandas as pd
data = pd.read_csv('Final_Output_After_Mapping.csv')
data.Closing_price = data.Closing_price.str[1:-1]
data.Opening_price = data.Opening_price.str[1:-1]
data.High_price = data.High_price.str[1:-1]
data.Low_price = data.Low_price.str[1:-1]

data.to_csv('Final_Output.csv')
