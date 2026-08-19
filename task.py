# Week 2: Data Collection, Cleaning and Preprocessing
# Logistics Data Analysis

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# --------------------------------------------------
# 1. DATA COLLECTION / LOAD DATA
# --------------------------------------------------

orders = pd.read_csv("data/olist_orders_dataset.csv")
items = pd.read_csv("data/olist_order_items_dataset.csv")
products = pd.read_csv("data/olist_products_dataset.csv")
customers = pd.read_csv("data/olist_customers_dataset.csv")

print("Data loaded successfully!")

# --------------------------------------------------
# 2. INITIAL DATA INSPECTION
# --------------------------------------------------

print("\n--- ORDERS DATA ---")
print(orders.head())

print("\nDataset Shape:")
print(orders.shape)

print("\nColumn Information:")
print(orders.info())

print("\nStatistical Summary:")
print(orders.describe(include="all"))

# --------------------------------------------------
# 3. CHECK MISSING VALUES
# --------------------------------------------------

print("\n--- MISSING VALUES ---")
print(orders.isnull().sum())

# --------------------------------------------------
# 4. REMOVE DUPLICATE RECORDS
# --------------------------------------------------

print("\nDuplicate rows before cleaning:",
      orders.duplicated().sum())

orders = orders.drop_duplicates()

print("Duplicate rows after cleaning:",
      orders.duplicated().sum())

# --------------------------------------------------
# 5. REMOVE RECORDS WITH MISSING ORDER ID
# --------------------------------------------------

orders = orders.dropna(subset=["order_id"])

# --------------------------------------------------
# 6. CONVERT DATE COLUMNS
# --------------------------------------------------

date_columns = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]

for column in date_columns:
    orders[column] = pd.to_datetime(
        orders[column],
        errors="coerce"
    )

print("\nDate columns converted successfully.")

# --------------------------------------------------
# 7. CREATE DELIVERY DAYS
# --------------------------------------------------

orders["delivery_days"] = (
    orders["order_delivered_customer_date"]
    - orders["order_purchase_timestamp"]
).dt.total_seconds() / (24 * 60 * 60)

orders["delivery_days"] = orders["delivery_days"].round(2)

print("\nDelivery days created.")

# --------------------------------------------------
# 8. CHECK INVALID DELIVERY DAYS
# --------------------------------------------------

invalid_delivery = orders[
    orders["delivery_days"] < 0
]

print("\nInvalid delivery records:",
      len(invalid_delivery))

# Remove negative delivery times
orders.loc[
    orders["delivery_days"] < 0,
    "delivery_days"
] = None

# --------------------------------------------------
# 9. OUTLIER DETECTION USING IQR
# --------------------------------------------------

Q1 = orders["delivery_days"].quantile(0.25)
Q3 = orders["delivery_days"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

outliers = orders[
    (orders["delivery_days"] < lower_limit) |
    (orders["delivery_days"] > upper_limit)
]

print("\n--- OUTLIER ANALYSIS ---")
print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Limit:", lower_limit)
print("Upper Limit:", upper_limit)
print("Number of Outliers:", len(outliers))

# --------------------------------------------------
# 10. LOAD PRODUCT DATA
# --------------------------------------------------

print("\n--- PRODUCT DATA ---")

print(products.isnull().sum())

# Fill missing product category
products["product_category_name"] = (
    products["product_category_name"]
    .fillna("Unknown")
)

# Fill missing product weight with median
products["product_weight_g"] = (
    products["product_weight_g"]
    .fillna(
        products["product_weight_g"].median()
    )
)

# --------------------------------------------------
# 11. LOAD CUSTOMER DATA
# --------------------------------------------------

print("\n--- CUSTOMER DATA ---")

print(customers.head())

# Remove duplicate customers
customers = customers.drop_duplicates()

# --------------------------------------------------
# 12. MERGE DATA
# --------------------------------------------------

logistics_data = orders.merge(
    customers[
        [
            "customer_id",
            "customer_city",
            "customer_state"
        ]
    ],
    on="customer_id",
    how="left"
)

print("\nMerged logistics data:")
print(logistics_data.head())

# --------------------------------------------------
# 13. HANDLE MISSING VALUES
# --------------------------------------------------

print("\nMissing values after merging:")
print(logistics_data.isnull().sum())

# --------------------------------------------------
# 14. NORMALIZATION
# --------------------------------------------------

# Select columns that will be normalized
numeric_columns = [
    "delivery_days"
]

# Remove missing values temporarily for scaling
valid_rows = logistics_data[numeric_columns].notnull().all(axis=1)

scaler = MinMaxScaler()

logistics_data.loc[
    valid_rows,
    numeric_columns
] = scaler.fit_transform(
    logistics_data.loc[
        valid_rows,
        numeric_columns
    ]
)

print("\nNormalization completed.")

# --------------------------------------------------
# 15. FINAL DATA VALIDATION
# --------------------------------------------------

print("\n--- FINAL DATA VALIDATION ---")

print("Final Shape:")
print(logistics_data.shape)

print("\nMissing Values:")
print(logistics_data.isnull().sum())

print("\nDuplicate Rows:")
print(logistics_data.duplicated().sum())

print("\nFinal Dataset:")
print(logistics_data.head())

# --------------------------------------------------
# 16. SAVE CLEANED DATA
# --------------------------------------------------

logistics_data.to_csv(
    "data/cleaned_logistics_data.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")
print("File: data/cleaned_logistics_data.csv")