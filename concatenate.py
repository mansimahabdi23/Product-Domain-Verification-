import pandas as pd

flipkart_data = pd.read_csv("data/flipkart_cleaned_data.csv")
generated_data = pd.read_csv("data/generated_products.csv")

print("Flipkart columns:", flipkart_data.columns.tolist())
print("Generated data columns:", generated_data.columns.tolist())

combined_df = pd.concat([flipkart_data, generated_data], ignore_index=True)

# combined_df = combined_df.drop_duplicates(
#     subset=["product_name", "description", "product_category_tree"]
# )

combined_df.to_csv("data/combined_products.csv", index=False)
print("Combined dataset saved to data/combined_products.csv")
print("Total records in combined dataset:", len(combined_df))