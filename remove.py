import pandas as pd
df = pd.read_csv("data/flipkart_cleaned_data.csv")
df.drop(columns=["uniq_id"], inplace=True)
df.to_csv("flipkart_cleaned_data.csv", index=False)
