# Logistics Data Collection, Cleaning and Preprocessing

## 📌 Project Overview

This project demonstrates a complete **data collection, cleaning, and preprocessing pipeline for logistics analysis using Python**.

The project uses the **Brazilian E-Commerce Public Dataset by Olist** as a reference dataset. Logistics-related information such as orders, delivery dates, customer locations, and product information is processed to prepare the data for further analysis.

The main goal is to convert raw logistics data into a **clean, consistent, and analysis-ready dataset**.

---

## 🎯 Objectives

* Collect and load logistics-related data using Python.
* Understand the structure and characteristics of the dataset.
* Identify missing values and handle them appropriately.
* Remove duplicate records.
* Convert data into correct formats.
* Calculate delivery time.
* Detect outliers using the IQR method.
* Normalize numerical data using Min-Max Scaling.
* Validate and save the cleaned dataset.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Jupyter Notebook / VS Code**

---

## 📂 Project Structure

```text
logistics-data-preprocessing/
│
├── data/
│   ├── olist_orders_dataset.csv
│   ├── olist_order_items_dataset.csv
│   ├── olist_products_dataset.csv
│   ├── olist_customers_dataset.csv
│   └── cleaned_logistics_data.csv
│
├── preprocessing.py
│
├── requirements.txt
│
├── README.md
│
└── docs/
    └── Week_2_Logistics_Data_Preprocessing_Report.docx
```

---

## 📊 Dataset

The project uses the **Brazilian E-Commerce Public Dataset by Olist**.

The dataset contains information related to:

* Orders
* Customers
* Products
* Sellers
* Payments
* Delivery dates
* Freight charges
* Product characteristics

For this project, the focus is mainly on information useful for **logistics and delivery analysis**.

---

## 🔄 Data Preprocessing Pipeline

The following steps are performed:

### 1. Data Collection

The required CSV files are loaded using Pandas.

```python
import pandas as pd

orders = pd.read_csv("data/olist_orders_dataset.csv")
```

### 2. Data Inspection

The dataset is inspected using:

```python
orders.head()
orders.info()
orders.describe()
orders.shape
```

This helps identify the number of records, columns, data types, and statistical characteristics.

### 3. Missing Value Handling

Missing values are identified using:

```python
orders.isnull().sum()
```

Depending on the column, missing values can be removed, replaced, or retained when the missing value has business meaning.

### 4. Duplicate Removal

Duplicate records are identified and removed:

```python
orders = orders.drop_duplicates()
```

This prevents the same order from being counted multiple times.

### 5. Data Type Conversion

Date columns are converted into proper datetime format:

```python
orders["order_purchase_timestamp"] = pd.to_datetime(
    orders["order_purchase_timestamp"],
    errors="coerce"
)
```

### 6. Delivery Time Calculation

Delivery duration is calculated using purchase and delivery timestamps:

```python
orders["delivery_days"] = (
    orders["order_delivered_customer_date"]
    - orders["order_purchase_timestamp"]
).dt.total_seconds() / (24 * 60 * 60)
```

This provides an important logistics KPI: **delivery time in days**.

### 7. Outlier Detection

The **Interquartile Range (IQR)** method is used to identify unusual delivery times.

```python
Q1 = orders["delivery_days"].quantile(0.25)
Q3 = orders["delivery_days"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
```

Outliers are investigated rather than automatically deleted because some unusually long deliveries may represent genuine logistics situations.

### 8. Normalization

Min-Max normalization is applied to numerical variables when required:

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

orders[["delivery_days"]] = scaler.fit_transform(
    orders[["delivery_days"]]
)
```

Normalization scales values between **0 and 1**, which can be useful for machine-learning applications.

### 9. Final Validation

The cleaned dataset is checked again:

```python
print(orders.shape)
print(orders.isnull().sum())
print(orders.duplicated().sum())
```

### 10. Export Cleaned Dataset

The final dataset is saved as:

```python
orders.to_csv(
    "data/cleaned_logistics_data.csv",
    index=False
)
```

---

## 📈 Expected Outcome

After preprocessing, the dataset becomes more suitable for:

* Delivery-time analysis
* Logistics KPI calculation
* Freight-cost analysis
* Customer location analysis
* Data visualization
* Exploratory Data Analysis
* Machine-learning models
* Delivery-delay prediction

---

## 💡 Key Learning

Data preprocessing is an important part of logistics analytics because poor-quality data can lead to incorrect conclusions.

For example, duplicate orders can increase the calculated order volume, missing delivery dates can affect average delivery time, and incorrect values can produce misleading logistics KPIs.

Therefore, **clean and reliable data is the foundation of accurate logistics analysis and decision-making.**

---

## ▶️ How to Run the Project

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/logistics-data-preprocessing.git
```

### Step 2: Open the project

```bash
cd logistics-data-preprocessing
```

### Step 3: Install required libraries

```bash
pip install -r requirements.txt
```

### Step 4: Place the dataset

Put the required CSV files inside the `data` folder.

### Step 5: Run the preprocessing script

```bash
python preprocessing.py
```

The cleaned dataset will be generated inside the `data` folder.

---

## 📄 Documentation

The complete methodology, explanations, Python code, data-quality issues, and reflection are documented in:

**`docs/Week_2_Logistics_Data_Preprocessing_Report.docx`**

---

## 👨‍💻 Author

**Ved Satpute**

B.Tech Computer Science & Engineering

---

## 📌 Project Type

**Academic Project – Week 2**

**Topic:** Data Collection, Cleaning, and Preprocessing for Logistics Analysis
