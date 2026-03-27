import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Supply Chain Dashboard", layout="wide")
st.title("📊 Supply Chain Analytics Dashboard")

# -----------------------------
# LOADING
# -----------------------------
with st.spinner("Loading..."):
    time.sleep(1)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("data.csv", encoding='latin1')

df.dropna(subset=['Sales', 'Profit'], inplace=True)
df.drop_duplicates(inplace=True)

df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Order Month'] = df['Order Date'].dt.month

# -----------------------------
# ANALYSIS
# -----------------------------
top_products = df.groupby('Product Name')['Sales'].sum().sort_values(ascending=False)
category_sales = df.groupby('Category')['Sales'].sum()
region_sales = df.groupby('Region')['Sales'].sum()
monthly_sales = df.groupby('Order Month')['Sales'].sum()

# -----------------------------
# INVENTORY
# -----------------------------
product_demand = top_products

np.random.seed(42)
stock_levels = pd.Series(
    np.random.randint(2000, 10000, size=len(product_demand)),
    index=product_demand.index
)

reorder_quantity = (product_demand - stock_levels).clip(lower=0)

# DEAD STOCK
low_sales_threshold = product_demand.mean() * 0.5

dead_stock = pd.DataFrame({
    "Product": product_demand.index,
    "Stock": stock_levels,
    "Sales": product_demand
})

dead_stock = dead_stock[
    (dead_stock["Sales"] < low_sales_threshold) &
    (dead_stock["Stock"] > stock_levels.mean())
].head(2)

# -----------------------------
# ML (optional kept)
# -----------------------------
X = np.array(monthly_sales.index).reshape(-1, 1)
y = monthly_sales.values

rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
rf_pred = rf.predict(X)
rf_future = rf.predict(np.array([13,14,15]).reshape(-1,1))

# -----------------------------
# SIDEBAR
# -----------------------------
option = st.sidebar.selectbox(
    "📌 Select Analysis",
    ["Full Dashboard", "Top Products", "Category",
     "Monthly Trend", "Region", "Inventory Insights", "Smart Reorder System"]
)

# -----------------------------
# FULL DASHBOARD
# -----------------------------
if option == "Full Dashboard":

    st.subheader("📌 Overall Insights")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Top Product", top_products.idxmax())
    col2.metric("Lowest Product", top_products.idxmin())
    col3.metric("Best Region", region_sales.idxmax())
    col4.metric("Worst Region", region_sales.idxmin())

    st.markdown("### 📊 Key Summary")
    st.success(f"""
    ✔ Top Product: {top_products.idxmax()}  
    ✔ Lowest Product: {top_products.idxmin()}  
    ✔ Peak Month: {monthly_sales.idxmax()}  
    ✔ Lowest Month: {monthly_sales.idxmin()}  
    ✔ Best Region: {region_sales.idxmax()}  
    ✔ Worst Region: {region_sales.idxmin()}  
    """)

    # ROW 1
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Products")
        fig, ax = plt.subplots()
        top_8 = top_products.head(8)[::-1]
        names = [n[:25]+"..." if len(n)>25 else n for n in top_8.index]
        ax.barh(names, top_8.values, color='#4e79a7')
        st.pyplot(fig)

    with col2:
        st.subheader("Category Distribution")
        fig, ax = plt.subplots()
        ax.pie(category_sales, labels=category_sales.index, autopct='%1.1f%%')
        st.pyplot(fig)

    # ROW 2
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Monthly Sales Trend")
        fig, ax = plt.subplots()
        ax.plot(monthly_sales, marker='o', color='#e15759')
        st.pyplot(fig)

    with col4:
        st.subheader("Region-wise Sales")
        fig, ax = plt.subplots()
        ax.bar(region_sales.index, region_sales.values, color='#76b7b2')
        st.pyplot(fig)

    # DATASET
    st.markdown("---")
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head(20))

# -----------------------------
# TOP PRODUCTS
# -----------------------------
elif option == "Top Products":

    fig, ax = plt.subplots()
    top_8 = top_products.head(8)[::-1]
    ax.barh(top_8.index, top_8.values)
    st.pyplot(fig)

    st.success(f"""
    ✔ Top Product: {top_products.idxmax()}  
    ✔ Lowest Product: {top_products.idxmin()}  
    """)

# -----------------------------
# CATEGORY
# -----------------------------
elif option == "Category":

    fig, ax = plt.subplots()
    ax.pie(category_sales, labels=category_sales.index, autopct='%1.1f%%')
    st.pyplot(fig)

    st.success(f"""
    ✔ Top Category: {category_sales.idxmax()}  
    ✔ Lowest Category: {category_sales.idxmin()}  
    """)

# -----------------------------
# MONTHLY
# -----------------------------
elif option == "Monthly Trend":

    fig, ax = plt.subplots()
    ax.plot(monthly_sales, marker='o')
    st.pyplot(fig)

    st.success(f"""
    ✔ Peak Month: {monthly_sales.idxmax()}  
    ✔ Lowest Month: {monthly_sales.idxmin()}  
    """)

# -----------------------------
# REGION
# -----------------------------
elif option == "Region":

    fig, ax = plt.subplots()
    ax.bar(region_sales.index, region_sales.values)
    st.pyplot(fig)

    st.success(f"""
    ✔ Best Region: {region_sales.idxmax()}  
    ✔ Worst Region: {region_sales.idxmin()}  
    """)

# -----------------------------
# INVENTORY
# -----------------------------
elif option == "Inventory Insights":

    st.subheader("Inventory Analysis")

    st.markdown("### High Demand")
    st.dataframe(product_demand.head(10))

    st.markdown("### Low Demand")
    st.dataframe(product_demand.tail(10))

    st.markdown("### Dead Stock")
    st.dataframe(dead_stock)

# -----------------------------
# REORDER SYSTEM
# -----------------------------
elif option == "Smart Reorder System":

    reorder_df = pd.DataFrame({
        "Product": product_demand.index,
        "Stock": stock_levels,
        "Demand": product_demand,
        "Shortage": reorder_quantity
    })

    reorder_df = reorder_df[reorder_df["Shortage"] > 0].head(8)

    fig, ax = plt.subplots()

    ax.barh(reorder_df["Product"], reorder_df["Stock"], label="Stock")
    ax.barh(reorder_df["Product"], reorder_df["Demand"], alpha=0.5, label="Demand")

    ax.legend()
    st.pyplot(fig)

    st.success(f"""
    ✔ Items needing reorder: {len(reorder_df)}  
    ✔ Highest shortage: {int(reorder_df['Shortage'].max())}  
    """)