#!/usr/bin/env python
# coding: utf-8

# # Overview
# The Dirty Cafe Sales dataset contains 10,000 rows of synthetic data representing sales transactions in a cafe. This dataset is intentionally "dirty," with missing values, inconsistent data, and errors introduced to provide a realistic scenario for data cleaning and exploratory data analysis (EDA). It can be used to practice cleaning techniques, data wrangling, and feature engineering.

# # Columns Description
# 1. Transaction ID : A unique identifier for each transaction. Always present and unique
# 2. Item : The name of the item purchased. May contain missing or invalid values
# 3. Quantity : The quantity of the item purchased. May contain missing or invalid values.
# 4. Price per Unit : The price of a single unit of the item. May contain missing or invalid values.
# 5. Total spent : The total amount spent on the transaction. Calculated as Quantity x Price per Unit
# 6. Payment Method : The method of payment used. May contain missing or invalid values
# 7. Location : The location where the transaction occured. May contain missing or invalid values.
# 8. Transaction date : The date of the transaction. May contain missing or invalid values 

# ## Import Package and Import Data

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


# In[2]:


os.getcwd()


# In[3]:


sales_data = pd.read_csv("cafe_sales.csv")
sales_data.head()


# In[4]:


display(sales_data.info(), 
        sales_data.duplicated().sum(), 
        sales_data.isna().sum())


# In[5]:


sales_data.nunique()


# # Cleansing Data
# 
# ## Fix the "Item" Columns

# In[6]:


sales_data = sales_data.replace(["ERROR", "UNKNOWN"], np.nan)


# In[7]:


sales_data.info()


# In[8]:


for i in sales_data.columns :
    print(f"{i} :")
    print(sales_data[i].unique())

print(sales_data.columns)


# In[9]:


sales_data["Quantity"] = pd.to_numeric(sales_data["Quantity"], errors = 'coerce')
sales_data["Price Per Unit"] = pd.to_numeric(sales_data["Price Per Unit"], errors = 'coerce')
sales_data["Total Spent"] = pd.to_numeric(sales_data["Total Spent"], errors = 'coerce')
sales_data["Transaction Date"] = pd.to_datetime(sales_data["Transaction Date"])


# In[10]:


sales_data.info()


# In[11]:


sales_data["Price Per Unit"] = sales_data["Price Per Unit"].fillna(sales_data["Total Spent"] / sales_data["Quantity"])


# In[12]:


sales_data["Price Per Unit"].info()


# In[13]:


conditions = [(sales_data["Item"].isna()) & (sales_data["Price Per Unit"] == 1.0),
              (sales_data["Item"].isna()) & (sales_data["Price Per Unit"] == 1.5),
              (sales_data["Item"].isna()) & (sales_data["Price Per Unit"] == 2.0),
              (sales_data["Item"].isna()) & (sales_data["Price Per Unit"] == 3.0),
              (sales_data["Item"].isna()) & (sales_data["Price Per Unit"] == 4.0),
              (sales_data["Item"].isna()) & (sales_data["Price Per Unit"] == 5.0)]

item = ["Cookie", "Tea", "Coffee", "Cake or Juice", "Smoothie or Sandwich", "Salad"]

sales_data ["Item"] = np.select(conditions, item, default = sales_data["Item"])


# In[14]:


sales_data["Item"] = sales_data["Item"].fillna(sales_data["Item"].mode()[0])


# In[15]:


sales_data.info()


# ## Fix the "Quantity", "Total Spent", "Price Per Unit" Columns

# In[16]:


sales_data["Quantity"] = sales_data["Quantity"].fillna(sales_data["Total Spent"] / sales_data["Price Per Unit"])
sales_data["Total Spent"] = sales_data["Total Spent"].fillna(sales_data["Quantity"] * sales_data["Price Per Unit"])


# In[17]:


sales_data.info()


# In[18]:


sales_data["Quantity"] = sales_data["Quantity"].fillna(sales_data["Quantity"].median())
sales_data["Price Per Unit"] = sales_data["Price Per Unit"].fillna(sales_data["Price Per Unit"].median())
sales_data["Total Spent"] = sales_data["Total Spent"].fillna(sales_data["Total Spent"].mean().round())


# In[19]:


sales_data.info()


# ## Fix the "Payment Method" and "Location" Columns

# In[20]:


for col in ["Payment Method", "Location"] :
    sales_data[col] = sales_data[col].fillna(sales_data[col].mode()[0])


# In[21]:


sales_data.info()


# ## Fix the "Transaction Date" Column

# In[22]:


sales_data["Transaction Date"] = sales_data["Transaction Date"].fillna(method='ffill')


# In[23]:


sales_data.info()


# ## Feature Engineering
# 
# Create new columns, such as Day of the Week or Transaction Month, for further analysis.

# In[24]:


sales_data["Day of the Week"] = sales_data["Transaction Date"].dt.day_name()
sales_data["Transaction Month"] = sales_data["Transaction Date"].dt.month_name()
sales_data["Transaction Year"] = sales_data["Transaction Date"].dt.year


# In[25]:


new_df = sales_data.sort_values("Transaction Date", ascending=True).reset_index(drop = True)


# In[26]:


pd.reset_option("display.max_rows", None)


# # Visualization

# In[27]:


new_df


# In[28]:


new_df.info()


# ## 1. Overall Sales Trend

# In[29]:


month = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

new_df["Transaction Month"] = pd.Categorical (new_df["Transaction Month"],
                                              categories = month,
                                              ordered = True)

quantity_months = new_df.groupby("Transaction Month", as_index=False)["Quantity"].sum()
quantity_months


# In[30]:


sns.barplot(quantity_months, x="Transaction Month", y="Quantity")
plt.xticks(rotation=45)
plt.ylim(2000)
plt.show()


# In[31]:


tren = new_df.groupby("Transaction Date", as_index=True)["Total Spent"].sum()


# In[32]:


# Cara lain menggunakan matplotlib
import matplotlib.dates as mdates

plt.figure(figsize=(12,6))
tren.plot(kind="line")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
plt.show()


# ## 2. Sales Trend on Weekend Days

# In[33]:


weekend = new_df[(new_df["Day of the Week"]=="Saturday") | (new_df["Day of the Week"]=="Sunday")]
weekend = weekend.groupby("Day of the Week", as_index=True)["Total Spent"].sum()


# In[34]:


weekend


# In[35]:


sns.barplot(weekend)
plt.ylim(12800,12850)
plt.show()


# In[36]:


print((weekend.value_counts(normalize=True)*100).round(2).astype(str)+'%')
weekend.plot(kind='pie', autopct='%0.0f%%', startangle=90, colors=plt.cm.Pastel1.colors)
plt.ylabel('')
plt.show()


# ## 3. Top Performing Products

# In[37]:


products = new_df.groupby("Item").agg(
item_number = ("Item", "count"),
quantity_sum = ("Quantity", "sum"),
quantity_price = ("Total Spent", "sum"))

products = products.reset_index()


# In[38]:


products


# In[39]:


top10_item = products.nlargest(10,"quantity_sum")
top10_price = products.nlargest(10,"quantity_price")


# In[40]:


plt.figure(figsize=(12,8))

plt.subplot(1, 2, 1)
sns.barplot(top10_item, x="item_number", y="Item", color='skyblue')
plt.title("Quantity of Item Purchased")

plt.subplot(1, 2, 2)
sns.barplot(top10_price, x="quantity_price", y="Item", color='pink')
plt.title("Price of Each Item Purchased")

plt.tight_layout()  # Agar layout tidak tumpang tindih
plt.show()


# ## 4. Payment Method Analysis 

# In[41]:


payment = (new_df["Payment Method"].value_counts(normalize=True)*100)
payment.plot(kind='pie', autopct='%2.2f%%', startangle=90, colors=plt.cm.Pastel2.colors)
plt.ylabel('')
plt.show()


# ## 5. Item Recommendation
# 
# Rekomendasi item yang sering dibeli bersamaan (Market Basket Analysis)
# dengan **Apriori Algorithm**

# In[42]:


new_df


# In[43]:


trolly = pd.pivot_table(new_df,
              index = "Transaction ID",
              columns = "Item", 
              values = "Quantity",
              aggfunc = "nunique", 
              fill_value = 0)

display(trolly, trolly.info())


# In[44]:


def encode(x) :
    if x == 0 :
        return False
    if x > 0 :
        return True
    
trolly_encode = trolly.applymap(encode)
display(trolly_encode, trolly_encode.info())


# In[45]:


from mlxtend.frequent_patterns import apriori

frequent_item = apriori(trolly_encode, min_support = .01, use_colnames = True).sort_values("support", ascending = False).reset_index(drop=True)
frequent_item["product_count"] = frequent_item["itemsets"].apply(lambda x : len(x))
frequent_item

