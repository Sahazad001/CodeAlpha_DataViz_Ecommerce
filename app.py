# Streamlit dashboard for Olist Operations leadership
# Run locally with: streamlit run app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import kagglehub

st.set_page_config(page_title="Olist Operations Dashboard", layout="wide")

# --- Load & prepare data (cached so it doesn't reload on every interaction) ---
@st.cache_data
def load_data():
    path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")
    orders = pd.read_csv(f"{path}/olist_orders_dataset.csv")
    order_items = pd.read_csv(f"{path}/olist_order_items_dataset.csv")
    customers = pd.read_csv(f"{path}/olist_customers_dataset.csv")
    payments = pd.read_csv(f"{path}/olist_order_payments_dataset.csv")
    reviews = pd.read_csv(f"{path}/olist_order_reviews_dataset.csv")
    products = pd.read_csv(f"{path}/olist_products_dataset.csv")
    sellers = pd.read_csv(f"{path}/olist_sellers_dataset.csv")
    category_translation = pd.read_csv(f"{path}/product_category_name_translation.csv")

    df = orders.merge(order_items, on='order_id', how='left')
    df = df.merge(customers, on='customer_id', how='left')
    df = df.merge(payments, on='order_id', how='left')
    df = df.merge(reviews, on='order_id', how='left')
    df = df.merge(products, on='product_id', how='left')
    df = df.merge(sellers, on='seller_id', how='left')
    df = df.merge(category_translation, on='product_category_name', how='left')

    date_cols = ['order_purchase_timestamp', 'order_delivered_customer_date', 'order_estimated_delivery_date']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    df['delivery_delay_days'] = (df['order_delivered_customer_date'] - df['order_estimated_delivery_date']).dt.days
    df['delivery_time_days'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days
    df['product_category_name_english'] = df['product_category_name_english'].fillna('unknown')
    df = df[df['order_purchase_timestamp'] < '2018-08-01']  # exclude incomplete trailing months

    return df

df = load_data()

# --- Header ---
st.title("📦 Olist Operations Dashboard")
st.markdown("**Built for:** Head of Operations | **Purpose:** Track delivery performance and customer satisfaction drivers")

# --- Sidebar Filters ---
st.sidebar.header("Filters")
states = st.sidebar.multiselect("Select States", options=sorted(df['customer_state'].dropna().unique()), default=None)
if states:
    df = df[df['customer_state'].isin(states)]

# --- KPI Row ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("On-Time Delivery Rate", f"{(df['delivery_delay_days'] <= 0).mean()*100:.1f}%")
col2.metric("Avg Review Score", f"{df['review_score'].mean():.2f} / 5")
col3.metric("Avg Delivery Time", f"{df['delivery_time_days'].mean():.1f} days")
col4.metric("1-Star Review Rate", f"{(df['review_score']==1).mean()*100:.1f}%")

st.divider()

# --- Chart 1: Delivery trend over time ---
st.subheader("📈 On-Time Delivery Rate Trend")
monthly_perf = df.groupby(df['order_purchase_timestamp'].dt.to_period('M').astype(str)).apply(
    lambda x: (x['delivery_delay_days'] <= 0).mean() * 100
).reset_index(name='on_time_rate')
fig1 = px.line(monthly_perf, x='order_purchase_timestamp', y='on_time_rate', markers=True,
               labels={'order_purchase_timestamp': 'Month', 'on_time_rate': 'On-Time Rate (%)'})
st.plotly_chart(fig1, use_container_width=True)

# --- Chart 2: Delivery by state ---
st.subheader("🗺️ Delivery Time by State")
state_data = df.groupby('customer_state')['delivery_time_days'].mean().sort_values(ascending=False).reset_index()
fig2 = px.bar(state_data, x='delivery_time_days', y='customer_state', orientation='h',
              labels={'delivery_time_days': 'Avg Delivery Time (days)', 'customer_state': 'State'})
st.plotly_chart(fig2, use_container_width=True)

# --- Chart 3: Category satisfaction ---
st.subheader("📦 Category Review Scores")
cat_data = df.groupby('product_category_name_english').agg(
    avg_review=('review_score', 'mean'), count=('order_id', 'count')
)
cat_data = cat_data[cat_data['count'] > 100].sort_values('avg_review').head(10).reset_index()
fig3 = px.bar(cat_data, x='avg_review', y='product_category_name_english', orientation='h',
              labels={'avg_review': 'Avg Review Score', 'product_category_name_english': 'Category'})
st.plotly_chart(fig3, use_container_width=True)

# --- Chart 4: Delay vs Review correlation ---
st.subheader("⏱️ Delivery Delay Impact on Reviews")
delay_review = df.groupby('review_score')['delivery_delay_days'].mean().reset_index()
fig4 = px.bar(delay_review, x='review_score', y='delivery_delay_days',
              labels={'review_score': 'Review Score', 'delivery_delay_days': 'Avg Delivery Delay (days)'})
st.plotly_chart(fig4, use_container_width=True)

st.divider()
st.caption("Data: Olist Brazilian E-Commerce Dataset (2016–Jul 2018) | Built as part of CodeAlpha Data Analytics Internship")
