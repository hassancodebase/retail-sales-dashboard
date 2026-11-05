from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Retail Sales Dashboard", page_icon="📈", layout="wide")

CSV_PATH = Path(__file__).parent / "mock_kaggle.csv"
COLUMN_ALIASES = {
    "date": ["data", "date", "dt", "day"],
    "sales": ["venda", "sales", "sale", "qty", "quantity"],
    "stock": ["estoque", "stock", "inventory"],
    "price": ["preco", "preço", "price", "unit_price"],
}

def _rename_columns(df):
    lower = {c.lower().strip(): c for c in df.columns}
    return df.rename(columns={lower[a]: std for std, aliases in COLUMN_ALIASES.items() for a in aliases if a in lower})

def _synthetic_sample(n=900):
    rng = np.random.default_rng(42)
    dates = pd.date_range("2014-01-01", periods=n, freq="D")
    t, sales = np.arange(n), np.clip(30 * (1 + 0.25 * np.isin(dates.dayofweek, [4, 5])) * (1 + 0.2 * np.sin(2 * np.pi * np.arange(n) / 365.25)) + rng.normal(0, 4, n), 0, None).round()
    stock, level = np.empty(n), 300.0
    for i in range(n):
        level = 300.0 if (level - sales[i]) < 60 else level - sales[i]
        stock[i] = max(level, 0)
    return pd.DataFrame({"date": dates, "sales": sales, "stock": stock.round(), "price": (10 + 0.8 * np.sin(2 * np.pi * t / 120) + rng.normal(0, 0.3, n)).round(2)})

@st.cache_data
def load_data():
    df = _rename_columns(pd.read_csv(CSV_PATH)) if CSV_PATH.exists() else _synthetic_sample()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    for c in ("sales", "stock", "price"): df[c] = pd.to_numeric(df[c], errors="coerce")
    df["revenue"] = df["sales"] * df["price"]
    df["year"], df["month"], df["month_name"] = df["date"].dt.year, df["date"].dt.month, df["date"].dt.strftime("%b")
    df["weekday"], df["weekday_num"], df["stockout"] = df["date"].dt.day_name(), df["date"].dt.dayofweek, df["stock"].fillna(0) <= 0
    return df, not CSV_PATH.exists()

df, using_sample = load_data()

# Sidebar Controls
st.sidebar.header("Controls")
min_d, max_d = df["date"].min().date(), df["date"].max().date()
start, end = st.sidebar.date_input("Date Range", (min_d, max_d), min_value=min_d, max_value=max_d)
sel_years = st.sidebar.multiselect("Years", sorted(df["year"].unique()), default=sorted(df["year"].unique()))
roll = st.sidebar.slider("Moving Average (days)", 1, 60, 14)
freq = {"Day": "D", "Week": "W", "Month": "MS"}[st.sidebar.radio("Aggregation", ["Day", "Week", "Month"], index=2)]

f = df[(df["date"].dt.date >= start) & (df["date"].dt.date <= end) & (df["year"].isin(sel_years))]

st.title("📈 Retail Sales Dashboard")
if using_sample: st.warning("Using synthetic demo data (mock_kaggle.csv not found).")
if f.empty: st.error("No data matches filters.") and st.stop()

# Key Metrics
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Units Sold", f"{f['sales'].sum():,.0f}")
c2.metric("Revenue", f"{f['revenue'].sum():,.0f}")
c3.metric("Daily Avg", f"{f['sales'].mean():,.1f}")
c4.metric("Avg Price", f"{f['price'].mean():,.2f}")
c5.metric("Stock-outs", f"{int(f['stockout'].sum())}", f"{f['stockout'].mean():.1%}", delta_color="off")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Trends", "📦 Distribution", "📅 Seasonality", "🔗 Relationships", "📋 Data"])

with tab1:
    d = f.set_index("date")
    agg = d["sales"].resample(freq).sum().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=agg["date"], y=agg["sales"], mode="lines", name="Sales", line=dict(width=2)))
    if freq == "D":
        fig.add_trace(go.Scatter(x=agg["date"], y=agg["sales"].rolling(roll).mean(), mode="lines", name=f"{roll}d MA", line=dict(width=3)))
    fig.update_layout(height=350, margin=dict(l=0, r=0, t=20, b=0), template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        rev = d["revenue"].resample(freq).sum().reset_index()
        st.plotly_chart(px.area(rev, x="date", y="revenue", title="Revenue Trend", height=300, template="plotly_white"), use_container_width=True)
    with right:
        m = d[["price", "stock"]].resample("MS").mean().reset_index()
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=m["date"], y=m["price"], name="Price"))
        fig2.add_trace(go.Bar(x=m["date"], y=m["stock"], name="Stock", yaxis="y2", opacity=0.4))
        fig2.update_layout(title="Price vs Stock Level", height=300, margin=dict(l=0, r=0, t=30, b=0), yaxis=dict(title="Price"), yaxis2=dict(title="Stock", overlaying="y", side="right"), template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    a, b = st.columns(2)
    with a:
        st.plotly_chart(px.histogram(f, x="sales", nbins=40, marginal="box", title="Sales Distribution", height=360, template="plotly_white"), use_container_width=True)
    with b:
        st.plotly_chart(px.box(f, x="year", y="sales", title="Sales by Year", height=360, template="plotly_white"), use_container_width=True)
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    st.plotly_chart(px.violin(f, x="weekday", y="sales", box=True, category_orders={"weekday": order}, title="Sales by Weekday", height=340, template="plotly_white"), use_container_width=True)

with tab3:
    a, b = st.columns(2)
    with a:
        w = f.groupby(["weekday_num", "weekday"], as_index=False)["sales"].mean()
        st.plotly_chart(px.bar(w, x="weekday", y="sales", color="sales", title="Avg Sales by Weekday", height=320, template="plotly_white"), use_container_width=True)
    with b:
        mo = f.groupby(["month", "month_name"], as_index=False)["sales"].mean()
        st.plotly_chart(px.bar(mo, x="month_name", y="sales", color="sales", title="Avg Sales by Month", height=320, template="plotly_white"), use_container_width=True)
    
    pivot = f.pivot_table(index="weekday_num", columns="month", values="sales", aggfunc="mean")
    pivot.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    st.plotly_chart(px.imshow(pivot, aspect="auto", color_continuous_scale="YlOrRd", title="Weekday vs Month Heatmap", height=340, template="plotly_white"), use_container_width=True)

with tab4:
    a, b = st.columns(2)
    with a:
        try:
            fig = px.scatter(f, x="price", y="sales", opacity=0.4, trendline="ols", title="Price Elasticity", height=360, template="plotly_white")
        except:
            fig = px.scatter(f, x="price", y="sales", opacity=0.4, title="Price Elasticity", height=360, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    with b:
        corr = f[["sales", "stock", "price", "revenue"]].corr()
        st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Correlation Matrix", height=360, template="plotly_white"), use_container_width=True)

with tab5:
    st.dataframe(f[["date", "sales", "stock", "price", "revenue"]], use_container_width=True, height=320)
    st.download_button("Download CSV", f.to_csv(index=False).encode(), "filtered_sales.csv", "text/csv")