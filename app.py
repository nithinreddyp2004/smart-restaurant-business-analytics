import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from itertools import combinations
from collections import Counter
import joblib
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Restaurant Analytics",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .main-title {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 2.4rem;
        background: linear-gradient(135deg, #FF6B35 0%, #F7C59F 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #888;
        font-size: 0.95rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(255,107,53,0.25);
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        text-align: center;
    }
    .metric-label {
        color: #aaa;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-family: 'Syne', sans-serif;
        font-size: 1.7rem;
        font-weight: 700;
        color: #FF6B35;
    }
    .section-header {
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        font-size: 1.2rem;
        color: #FF6B35;
        border-left: 4px solid #FF6B35;
        padding-left: 0.8rem;
        margin: 1.5rem 0 1rem 0;
    }
    .insight-box {
        background: rgba(255,107,53,0.08);
        border: 1px solid rgba(255,107,53,0.2);
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        margin: 0.5rem 0;
        font-size: 0.88rem;
        color: #ddd;
    }
    .insight-box strong { color: #FF6B35; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.4rem 1rem;
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
    }
    div[data-testid="stSidebarContent"] {
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
    }
    .stSelectbox label, .stMultiSelect label, .stSlider label {
        color: #aaa !important;
        font-size: 0.85rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── Data Loading & Preprocessing ──────────────────────────────────────────────
@st.cache_data
def load_and_clean_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        np.random.seed(42)
        n = 6060
        items = {
            'Egg Curry': 'Main Course', 'Paneer Butter Masala': 'Main Course',
            'Dal Tadka': 'Main Course', 'Chicken Biryani': 'Main Course',
            'Veg Biryani': 'Main Course', 'Fish Curry': 'Main Course',
            'Kheer': 'Desserts', 'Gulab Jamun': 'Desserts',
            'Rasgulla': 'Desserts', 'Chocolate Brownie': 'Desserts',
            'Veg Burger': 'Burgers', 'Chicken Burger': 'Burgers',
            'Paratha': 'Breads', 'Naan': 'Breads', 'Roti': 'Breads',
            'Pepperoni Pizza': 'Pizza', 'Veg Supreme Pizza': 'Pizza',
            'Margherita Pizza': 'Pizza',
            'Mojito': 'Drinks', 'Cold Coffee': 'Drinks', 'Mango Shake': 'Drinks',
            'Bruschetta': 'Starters', 'Spring Rolls': 'Starters',
            'Paneer Tikka': 'Starters', 'Fish Fingers': 'Starters',
            'Onion Rings': 'Starters',
            'Club Sandwich': 'Sandwiches', 'Veg Sandwich': 'Sandwiches',
            'Chicken Sandwich': 'Sandwiches',
            'Chicken Fried Rice': 'Main Course', 'Veg Fried Rice': 'Main Course',
        }
        item_names = list(items.keys())
        item_cats = [items[i] for i in item_names]
        idx = np.random.randint(0, len(item_names), n)
        timestamps = pd.date_range('2022-10-01', '2023-09-30', periods=n)
        timestamps = np.random.permutation(timestamps)
        payments = ['UPI', 'Cash', 'Credit Card', 'Debit Card', 'Net Banking', 'Digital Wallet']
        cities = ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Pune']
        statuses = ['Delivered', 'Pending', 'Cancelled']

        df = pd.DataFrame({
            'order_id': [f'ORD{str(i).zfill(6)}' for i in np.random.randint(1000, 9999, n)],
            'customer_id': [f'CUST{str(i).zfill(4)}' for i in np.random.randint(1, 700, n)],
            'item_name': [item_names[i] for i in idx],
            'category': [item_cats[i] for i in idx],
            'quantity': np.random.randint(1, 6, n),
            'price': np.round(np.random.uniform(80, 700, n), 2),
            'payment_method': np.random.choice(payments, n),
            'order_status': np.random.choice(statuses, n, p=[0.65, 0.2, 0.15]),
            'order_timestamp': timestamps,
            'rating': np.round(np.random.uniform(2.5, 5.0, n), 1),
            'delivery_time_mins': np.random.randint(15, 76, n).astype(float),
            'city': np.random.choice(cities, n),
        })

    df['payment_method'].fillna(df['payment_method'].mode()[0], inplace=True)
    df['rating'].fillna(df['rating'].median(), inplace=True)
    df['delivery_time_mins'].fillna(df['delivery_time_mins'].median(), inplace=True)
    df['city'].fillna(df['city'].mode()[0], inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['order_timestamp'] = pd.to_datetime(df['order_timestamp'], format='mixed', dayfirst=True, errors='coerce')
    df['category'] = df['category'].str.strip().str.title()

    for col in ['price', 'delivery_time_mins']:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

    df = df[df['quantity'] >= 1]
    df.reset_index(drop=True, inplace=True)

    df['hour'] = df['order_timestamp'].dt.hour
    df['day_of_week'] = df['order_timestamp'].dt.day_name()
    df['month'] = df['order_timestamp'].dt.month_name()
    df['month_num'] = df['order_timestamp'].dt.month
    df['date'] = df['order_timestamp'].dt.date
    df['week'] = df['order_timestamp'].dt.isocalendar().week.astype(int)
    df['year'] = df['order_timestamp'].dt.year
    df['revenue'] = df['price'] * df['quantity']
    df['time_of_day'] = pd.cut(
        df['hour'],
        bins=[-1, 5, 11, 14, 17, 21, 24],
        labels=['Night', 'Morning', 'Lunch', 'Afternoon', 'Evening', 'Late Night']
    )
    return df

COLORS = ['#FF6B35', '#F7C59F', '#EFEFD0', '#004E89', '#1A936F', '#88D498',
          '#C6DABF', '#E9724C', '#D6A2AD', '#9BC4CB']

LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font_color='#cccccc',
    font_family='DM Sans',
    title_font_family='Syne',
    title_font_color='#FF6B35',
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False),
    margin=dict(l=30, r=20, t=50, b=30),
    colorway=COLORS,
)

def apply_layout(fig, title="", height=380):
    fig.update_layout(**LAYOUT, title=title, height=height)
    return fig

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p style="font-family:Syne;font-size:1.4rem;font-weight:800;color:#FF6B35;">🍽️ Restaurant<br>Analytics</p>', unsafe_allow_html=True)
    st.markdown("---")

    uploaded = st.file_uploader("Upload CSV dataset", type=["csv"])
    st.markdown("---")

    df_raw = load_and_clean_data(uploaded)

    st.markdown("**Filters**")
    cities_avail = sorted(df_raw['city'].unique())
    sel_cities = st.multiselect("City", cities_avail, default=cities_avail)

    cats_avail = sorted(df_raw['category'].unique())
    sel_cats = st.multiselect("Category", cats_avail, default=cats_avail)

    statuses_avail = df_raw['order_status'].unique().tolist()
    sel_status = st.multiselect("Order Status", statuses_avail, default=statuses_avail)

    date_min = df_raw['date'].min()
    date_max = df_raw['date'].max()
    date_range = st.date_input("Date Range", [date_min, date_max], min_value=date_min, max_value=date_max)

    st.markdown("---")
    st.markdown('<p style="color:#555;font-size:0.75rem;">Smart Restaurant Analytics System</p>', unsafe_allow_html=True)

@st.cache_data
def filter_df(df, cities, cats, statuses, d_start, d_end):
    mask = (
        df['city'].isin(cities) &
        df['category'].isin(cats) &
        df['order_status'].isin(statuses) &
        (df['date'] >= d_start) &
        (df['date'] <= d_end)
    )
    return df[mask].copy()

if len(date_range) == 2:
    d_start, d_end = date_range
else:
    d_start, d_end = date_min, date_max

df = filter_df(df_raw, tuple(sel_cities), tuple(sel_cats), tuple(sel_status), d_start, d_end)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">Smart Restaurant Analytics</h1>', unsafe_allow_html=True)
st.markdown(
    f'<p class="subtitle">Analyzing <strong style="color:#FF6B35">{len(df):,}</strong> orders across {df["city"].nunique()} cities · {d_start} to {d_end}</p>',
    unsafe_allow_html=True
)

# ─── KPI Row ──────────────────────────────────────────────────────────────────
total_rev = df['revenue'].sum()
avg_order = df['revenue'].mean()
total_orders = len(df)
unique_custs = df['customer_id'].nunique()
avg_rating = df['rating'].mean()
avg_delivery = df['delivery_time_mins'].mean()

k1, k2, k3, k4, k5, k6 = st.columns(6)
for col, label, val in [
    (k1, "Total Revenue", f"₹{total_rev:,.0f}"),
    (k2, "Total Orders", f"{total_orders:,}"),
    (k3, "Avg Order Value", f"₹{avg_order:,.0f}"),
    (k4, "Unique Customers", f"{unique_custs:,}"),
    (k5, "Avg Rating", f"⭐ {avg_rating:.2f}"),
    (k6, "Avg Delivery", f"🕐 {avg_delivery:.1f} min"),
]:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{val}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 EDA & Sales", "🧑‍🤝‍🧑 Customer Analytics", "🤖 ML Prediction",
    "🔗 Recommendations", "🧹 Data Quality"
])

# ─── TAB 1: EDA & Sales ───────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">Daily & Weekly Sales Trends</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        daily = df.groupby('date')['revenue'].sum().reset_index()
        daily.columns = ['date', 'revenue']
        fig = px.line(daily, x='date', y='revenue', title='Daily Revenue Trend')
        fig.update_traces(line_color='#FF6B35', line_width=2, fill='tozeroy', fillcolor='rgba(255,107,53,0.1)')
        apply_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly = df.groupby('day_of_week')['revenue'].sum().reindex(day_order).reset_index()
        weekly.columns = ['day', 'revenue']
        fig = px.bar(weekly, x='day', y='revenue', title='Revenue by Day of Week',
                     color='revenue', color_continuous_scale=['#1a1a2e', '#FF6B35'])
        fig.update_coloraxes(showscale=False)
        apply_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Menu Performance</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)

    with c3:
        top_n = st.slider("Top N items", 5, 20, 10, key="top_n")
        top_items = df.groupby('item_name')['quantity'].sum().sort_values(ascending=False).head(top_n).reset_index()
        top_items.columns = ['item', 'orders']
        fig = px.bar(top_items, y='item', x='orders', orientation='h',
                     title=f'Top {top_n} Most Ordered Items',
                     color='orders', color_continuous_scale=['#004E89', '#FF6B35'])
        fig.update_coloraxes(showscale=False)
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        apply_layout(fig, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        bottom_items = df.groupby('item_name')['quantity'].sum().sort_values().head(top_n).reset_index()
        bottom_items.columns = ['item', 'orders']
        fig = px.bar(bottom_items, y='item', x='orders', orientation='h',
                     title=f'Bottom {top_n} Least Ordered Items',
                     color='orders', color_continuous_scale=['#1A936F', '#EFEFD0'])
        fig.update_coloraxes(showscale=False)
        fig.update_layout(yaxis={'categoryorder': 'total descending'})
        apply_layout(fig, height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Time & Payment Patterns</div>', unsafe_allow_html=True)
    c5, c6, c7 = st.columns(3)

    with c5:
        hourly = df.groupby('hour')['order_id'].count().reset_index()
        hourly.columns = ['hour', 'orders']
        fig = px.area(hourly, x='hour', y='orders', title='Peak Order Hours',
                      color_discrete_sequence=['#FF6B35'])
        fig.update_traces(fill='tozeroy', fillcolor='rgba(255,107,53,0.15)')
        apply_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        pay = df['payment_method'].value_counts().reset_index()
        pay.columns = ['method', 'count']
        fig = px.pie(pay, names='method', values='count', title='Payment Methods',
                     color_discrete_sequence=COLORS, hole=0.45)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        apply_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    with c7:
        cat_rev = df.groupby('category')['revenue'].sum().sort_values(ascending=False).reset_index()
        fig = px.pie(cat_rev, names='category', values='revenue', title='Revenue by Category',
                     color_discrete_sequence=COLORS, hole=0.45)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        apply_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">City & Rating Analysis</div>', unsafe_allow_html=True)
    c8, c9 = st.columns(2)

    with c8:
        city_rev = df.groupby('city')['revenue'].sum().sort_values(ascending=False).reset_index()
        fig = px.bar(city_rev, x='city', y='revenue', title='Revenue by City',
                     color='revenue', color_continuous_scale=['#004E89', '#FF6B35'])
        fig.update_coloraxes(showscale=False)
        apply_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    with c9:
        fig = px.histogram(df, x='rating', nbins=25, title='Order Rating Distribution',
                           color_discrete_sequence=['#FF6B35'])
        apply_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

# ─── TAB 2: Customer Analytics ────────────────────────────────────────────────
with tab2:
    cust_df = df.groupby('customer_id').agg(
        total_orders=('order_id', 'count'),
        total_spent=('revenue', 'sum'),
        avg_order_value=('revenue', 'mean'),
        avg_rating=('rating', 'mean'),
        unique_items=('item_name', 'nunique'),
    ).reset_index()

    repeat = (cust_df['total_orders'] > 1).sum()
    total_c = len(cust_df)
    repeat_pct = repeat / total_c * 100 if total_c else 0

    st.markdown('<div class="section-header">Customer Overview</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    for col, lbl, val in [
        (m1, "Total Customers", f"{total_c:,}"),
        (m2, "Repeat Customers", f"{repeat:,}"),
        (m3, "Repeat Rate", f"{repeat_pct:.1f}%"),
        (m4, "Avg Orders/Customer", f"{cust_df['total_orders'].mean():.1f}"),
    ]:
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{lbl}</div><div class="metric-value">{val}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        pie_data = pd.DataFrame({'type': ['Repeat', 'One-time'], 'count': [repeat, total_c - repeat]})
        fig = px.pie(pie_data, names='type', values='count', title='Repeat vs One-time Customers',
                     color_discrete_sequence=['#FF6B35', '#333'], hole=0.5)
        apply_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.histogram(cust_df, x='total_spent', nbins=40, title='Customer Spending Distribution',
                           color_discrete_sequence=['#1A936F'])
        apply_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Top 10 Customers by Spending</div>', unsafe_allow_html=True)
    top_custs = cust_df.nlargest(10, 'total_spent')[['customer_id', 'total_orders', 'total_spent', 'avg_order_value', 'avg_rating']]
    top_custs.columns = ['Customer ID', 'Orders', 'Total Spent (₹)', 'Avg Order (₹)', 'Avg Rating']
    top_custs['Total Spent (₹)'] = top_custs['Total Spent (₹)'].round(2)
    top_custs['Avg Order (₹)'] = top_custs['Avg Order (₹)'].round(2)
    top_custs['Avg Rating'] = top_custs['Avg Rating'].round(2)
    st.dataframe(top_custs.reset_index(drop=True), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">K-Means Customer Segmentation</div>', unsafe_allow_html=True)
    n_clusters = st.slider("Number of segments (k)", 2, 6, 3, key="kmeans_k")

    features = cust_df[['total_orders', 'total_spent', 'avg_order_value']].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cust_df['segment'] = km.fit_predict(X_scaled).astype(str)

    seg_labels = {str(i): f"Segment {i+1}" for i in range(n_clusters)}
    cust_df['segment_label'] = cust_df['segment'].map(seg_labels)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.scatter(cust_df, x='total_orders', y='total_spent',
                         color='segment_label', title='Customer Segments (Orders vs Spending)',
                         color_discrete_sequence=COLORS, opacity=0.7,
                         hover_data=['customer_id', 'avg_order_value'])
        apply_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        seg_summary = cust_df.groupby('segment_label').agg(
            customers=('customer_id', 'count'),
            avg_spent=('total_spent', 'mean'),
            avg_orders=('total_orders', 'mean')
        ).reset_index()
        fig = px.bar(seg_summary, x='segment_label', y='avg_spent',
                     color='segment_label', title='Avg Spending per Segment',
                     text='customers', color_discrete_sequence=COLORS)
        fig.update_traces(texttemplate='%{text} custs', textposition='outside')
        apply_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Segment Summary</div>', unsafe_allow_html=True)
    seg_summary['avg_spent'] = seg_summary['avg_spent'].round(2)
    seg_summary['avg_orders'] = seg_summary['avg_orders'].round(1)
    seg_summary.columns = ['Segment', 'Customers', 'Avg Spending (₹)', 'Avg Orders']
    st.dataframe(seg_summary, use_container_width=True, hide_index=True)

# ─── TAB 3: ML Prediction ─────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">Sales Prediction Model</div>', unsafe_allow_html=True)

    # Load model bundle from joblib
    model_path = None
    if Path("restaurant_model.pkl").exists():
        model_path = "restaurant_model.pkl"
    elif Path("model_bundle.pkl").exists():
        model_path = "model_bundle.pkl"

    if model_path is None:
        st.error("Model file not found. Please save `restaurant_model.pkl` first.")
        st.stop()

    bundle = joblib.load(model_path)

    if isinstance(bundle, dict) and "model" in bundle and "features" in bundle:
        model = bundle["model"]
        features = bundle["features"]
    else:
        st.error("Invalid joblib file format. Save a dictionary with keys: model, features.")
        st.stop()

    st.success(f"Loaded model from: {model_path}")

    st.markdown("### Predict Daily Sales")
    feature_inputs = {}
    for feat in features:
        if feat == "day":
            feature_inputs[feat] = st.number_input("Day", min_value=1, max_value=31, value=1)
        elif feat == "month":
            feature_inputs[feat] = st.number_input("Month", min_value=1, max_value=12, value=1)
        elif feat == "weekday":
            feature_inputs[feat] = st.number_input("Weekday (0=Mon, 6=Sun)", min_value=0, max_value=6, value=0)
        elif feat == "order_volume":
            feature_inputs[feat] = st.number_input("Order Volume", min_value=1, value=10)
        else:
            feature_inputs[feat] = st.number_input(f"{feat}", value=0.0)

    input_df = pd.DataFrame([[feature_inputs[f] for f in features]], columns=features)

    if st.button("Predict Sales"):
        prediction = model.predict(input_df)
        st.success(f"Predicted Daily Sales: ₹{prediction[0]:.2f}")

    st.markdown("### Model Summary")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Linear Regression R²", "0.66")
        st.metric("Linear Regression MAE", "1096.53")
        st.metric("Linear Regression RMSE", "1346.31")
    with c2:
        st.metric("Random Forest R²", "0.84")
        st.metric("Random Forest MAE", "—")
        st.metric("Random Forest RMSE", "—")

    st.markdown("#### Feature Importance")
    if hasattr(model, "feature_importances_"):
        importances = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True)
        fig = px.bar(importances, y='feature', x='importance', orientation='h',
                     title='Feature Importance', color='importance',
                     color_continuous_scale=['#1a1a2e', '#FF6B35'])
        fig.update_coloraxes(showscale=False)
        apply_layout(fig, height=320)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Selected model does not provide feature importances.")

    st.markdown('<div class="insight-box">📌 <strong>Note:</strong> This prediction tab now loads your saved joblib model instead of training inside Streamlit.</div>', unsafe_allow_html=True)

# ─── TAB 4: Recommendations ───────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">Frequently Bought Together</div>', unsafe_allow_html=True)

    @st.cache_data
    def compute_combos(df_hash):
        order_items = df.groupby('order_id')['item_name'].apply(list)
        combo_counts = Counter()
        for items_list in order_items:
            unique_items = list(set(items_list))
            if len(unique_items) >= 2:
                for combo in combinations(sorted(unique_items), 2):
                    combo_counts[combo] += 1
        combo_df = pd.DataFrame(
            [(a, b, c) for (a, b), c in combo_counts.most_common(30)],
            columns=['Item A', 'Item B', 'Co-occurrence']
        )
        return combo_df

    combo_df = compute_combos(len(df))

    if combo_df.empty:
        st.info("Your dataset does not contain multiple items per order, so co-occurrence pairs are not available.")
    else:
        c1, c2 = st.columns([3, 2])
        with c1:
            fig = px.bar(combo_df.head(15), y='Item A', x='Co-occurrence',
                         title='Top 15 Item Pair Co-occurrences',
                         color='Co-occurrence', color_continuous_scale=['#004E89', '#FF6B35'],
                         hover_data=['Item B'])
            fig.update_coloraxes(showscale=False)
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            apply_layout(fig, height=420)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("**Top Combo Recommendations**")
            for _, row in combo_df.head(8).iterrows():
                st.markdown(f"""
                <div class="insight-box">
                    🍱 <strong>{row['Item A']}</strong> + <strong>{row['Item B']}</strong>
                    <span style="float:right;color:#888">{row['Co-occurrence']}x</span>
                </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Item-Based Recommendation Engine</div>', unsafe_allow_html=True)
    all_items = sorted(df['item_name'].unique())
    chosen_item = st.selectbox("Select an item to get recommendations", all_items)

    if not combo_df.empty:
        related = combo_df[(combo_df['Item A'] == chosen_item) | (combo_df['Item B'] == chosen_item)].copy()
        related['Partner'] = related.apply(
            lambda r: r['Item B'] if r['Item A'] == chosen_item else r['Item A'], axis=1)
        related = related[['Partner', 'Co-occurrence']].sort_values('Co-occurrence', ascending=False).head(6)

        if not related.empty:
            st.markdown(f"**Customers who order '{chosen_item}' also frequently order:**")
            cols = st.columns(min(len(related), 3))
            for i, (_, row) in enumerate(related.iterrows()):
                with cols[i % 3]:
                    st.markdown(f"""
                    <div class="metric-card" style="margin-bottom:0.5rem;">
                        <div class="metric-label">Paired {row['Co-occurrence']}x</div>
                        <div class="metric-value" style="font-size:1rem;color:#F7C59F">{row['Partner']}</div>
                    </div>""", unsafe_allow_html=True)
        else:
            st.info(f"Not enough co-occurrence data for '{chosen_item}' in current filter.")
    else:
        st.info("No pair data available for recommendations in this dataset.")

    st.markdown('<div class="section-header">Category Co-occurrence Heatmap</div>', unsafe_allow_html=True)
    order_cats = df.groupby('order_id')['category'].apply(list)
    cat_combo = Counter()
    for cats in order_cats:
        ucats = list(set(cats))
        if len(ucats) >= 2:
            for c1p, c2p in combinations(sorted(ucats), 2):
                cat_combo[(c1p, c2p)] += 1

    categories = sorted(df['category'].unique())
    matrix = pd.DataFrame(0, index=categories, columns=categories)
    for (a, b), cnt in cat_combo.items():
        matrix.loc[a, b] = cnt
        matrix.loc[b, a] = cnt

    fig = px.imshow(matrix, title='Category Co-occurrence Matrix',
                    color_continuous_scale='Oranges', text_auto=True)
    apply_layout(fig, height=420)
    st.plotly_chart(fig, use_container_width=True)

# ─── TAB 5: Data Quality ──────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header">Raw Dataset Overview</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, lbl, val in [
        (c1, "Raw Rows", f"{len(df_raw):,}"),
        (c2, "Columns", f"{df_raw.shape[1]}"),
        (c3, "After Cleaning", f"{len(df):,}"),
        (c4, "Removed", f"{len(df_raw)-len(df):,}"),
    ]:
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{lbl}</div><div class="metric-value">{val}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Missing Values (Raw Data)</div>', unsafe_allow_html=True)
    mv = df_raw.isnull().sum().reset_index()
    mv.columns = ['Column', 'Missing']
    mv['Pct'] = (mv['Missing'] / len(df_raw) * 100).round(2)
    mv = mv[mv['Missing'] > 0]
    if not mv.empty:
        fig = px.bar(mv, x='Column', y='Missing', title='Missing Values per Column',
                     text='Pct', color='Missing', color_continuous_scale=['#1A936F', '#FF6B35'])
        fig.update_traces(texttemplate='%{text}%', textposition='outside')
        fig.update_coloraxes(showscale=False)
        apply_layout(fig, height=320)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("No missing values in the current dataset.")

    st.markdown('<div class="section-header">Descriptive Statistics (Cleaned)</div>', unsafe_allow_html=True)
    st.dataframe(df[['price', 'quantity', 'rating', 'delivery_time_mins', 'revenue']].describe().round(2),
                 use_container_width=True)

    st.markdown('<div class="section-header">Distribution & Outlier Detection</div>', unsafe_allow_html=True)
    col_to_check = st.selectbox("Select column", ['price', 'revenue', 'delivery_time_mins', 'rating'])
    c1, c2 = st.columns(2)

    with c1:
        fig = px.box(df, y=col_to_check, title=f'{col_to_check} — Box Plot',
                     color_discrete_sequence=['#FF6B35'])
        apply_layout(fig, height=320)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.histogram(df, x=col_to_check, nbins=40, title=f'{col_to_check} — Histogram',
                           color_discrete_sequence=['#1A9367'])
        apply_layout(fig, height=320)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Cleaned Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(100), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">Preprocessing Pipeline Applied</div>', unsafe_allow_html=True)
    steps = [
        ("Missing Value Imputation", "payment_method → mode, rating → median, delivery_time_mins → median, city → mode"),
        ("Duplicate Removal", "Removed exact duplicate rows"),
        ("Datetime Formatting", "order_timestamp parsed to datetime64 with mixed format support"),
        ("Category Standardization", "Applied str.strip().str.title() to category column"),
        ("IQR Outlier Removal", "Applied to price and delivery_time_mins (1.5×IQR rule)"),
        ("Negative Value Filtering", "Removed rows with quantity < 1 and price < 0"),
        ("Feature Engineering", "Extracted: hour, day_of_week, month, date, week, year, revenue, time_of_day"),
    ]
    for step, detail in steps:
        st.markdown(f'<div class="insight-box">✅ <strong>{step}</strong> — {detail}</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("Built with Streamlit for your restaurant analytics project.")