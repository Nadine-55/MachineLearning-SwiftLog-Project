"""SwiftLog prediction app. Run from the project root: streamlit run App/app.py"""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "Models"
DATA = ROOT / "Data" / "Processed_Datasets"
CITIES = ["Abuja", "Lagos", "Ibadan", "Kano"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

st.set_page_config(page_title="SwiftLog Predictions", layout="wide")


@st.cache_resource
def load_bundle():
    return joblib.load(MODELS / "Best_Model.pkl")


@st.cache_data
def load_tables():
    return (pd.read_csv(DATA / "demand_features.csv"),
            pd.read_csv(DATA / "customer_features.csv"),
            pd.read_csv(DATA / "warehouse_features.csv"),
            pd.read_csv(MODELS / "final_delay.csv"),
            pd.read_csv(MODELS / "final_regression.csv"))


bundle = load_bundle()
demand, cust, wh, res_delay, res_reg = load_tables()
WH_CITY = wh.drop_duplicates("WarehouseID").set_index("WarehouseID")["WarehouseCity"].to_dict()


def banner(problem):
    if problem == "delay":
        t = res_delay.iloc[0]
        st.warning(f"Reliability check: on unseen test data this model scored ROC-AUC "
                   f"{t['ROC_AUC']:.2f} (0.50 is a coin flip) and accuracy {t['Accuracy']:.2f}, "
                   f"the same as always predicting 'Delayed'. Model confidence: LOW. "
                   f"Use as a demonstration, not for decisions.")
    else:
        g = res_reg[res_reg["Problem"] == problem]
        tuned, base = g.iloc[0], g.iloc[1]
        st.warning( f"The model does not beat the average. Model confidence: LOW. "
                   f"Use as a demonstration, not for decisions.")


def regression_output(problem, pred, label, fmt):
    rmse = res_reg[res_reg["Problem"] == problem].iloc[0]["RMSE"]
    st.metric(label, fmt.format(pred))
    st.write(f"Typical error band (plus or minus one test RMSE): "
             f"{fmt.format(max(pred - rmse, 0))} to {fmt.format(pred + rmse)}")


st.title("SwiftLog Logistics Predictions")
st.caption("Each tab uses one model. Every tab shows how "
           "reliable that model was on unseen data.")
tab1, tab2, tab3, tab4 = st.tabs(["Delivery delay", "Demand forecast",
                                  "Customer value", "Warehouse performance"])

#  Delivery delay
with tab1:
    banner("delay")
    m = bundle["delay"]
    c1, c2, c3 = st.columns(3)
    with c1:
        wid = st.selectbox("Dispatch warehouse", sorted(WH_CITY), key="d_wid")
        ccity = st.selectbox("Customer city", CITIES, key="d_city")
        month = st.slider("Order month", 1, 12, 9, key="d_month")
        dow = st.selectbox("Day of week", range(7), format_func=lambda i: DAYS[i], key="d_dow")
    with c2:
        n_lines = st.number_input("Order lines", 0, 15, 3, key="d_lines")
        total_qty = st.number_input("Total quantity", 0, 100, 17, key="d_qty")
        max_qty = st.number_input("Largest single-line quantity", 0, 10, 7, key="d_max")
        n_cat = st.number_input("Product categories in the order", 0, 3, 2, key="d_cat")
    with c3:
        value = st.number_input("Order value", 0.0, 25000.0, 3800.0, step=100.0, key="d_val")
        load7 = st.number_input("Shipments from this warehouse in the last 7 days", 0, 40, 9, key="d_load")

    row = pd.DataFrame([{
        "n_lines": n_lines, "total_qty": total_qty, "order_value": value,
        "n_categories": n_cat, "max_qty": max_qty, "has_items": int(n_lines > 0),
        "order_month": month, "order_dow": dow, "cross_city": int(ccity != WH_CITY[wid]),
        "wh_load_7d": load7, "WarehouseID": wid, "WarehouseCity": WH_CITY[wid],
        "CustomerCity": ccity}])
    if st.button("Predict delay risk", key="b_delay"):
        p = float(m["pipeline"].predict_proba(row[m["features"]])[0, 1])
        st.metric("Probability this shipment is delayed (3+ days)", f"{p:.0%}")
        st.progress(min(max(p, 0.0), 1.0))
        call = "Delayed" if p >= m["threshold"] else "On Time"
        st.write(f"Model call: **{call}** (decision threshold {m['threshold']:.2f}). "
                 f"For comparison, about 64% of all historical shipments were delayed.")

# Demand
with tab2:
    banner("demand")
    m = bundle["demand"]
    pid = st.selectbox("Product (forecasting September 2026)", sorted(demand["ProductID"].unique()), key="p_pid")
    pr = demand[demand["ProductID"] == pid]
    m8, m7 = pr[pr["month"] == 8].iloc[0], pr[pr["month"] == 7].iloc[0]
    c1, c2, c3 = st.columns(3)
    with c1:
        lag1 = st.number_input("Units sold last month", 0.0, 500.0, float(m8["units"]), key=f"p_l1{pid}")
        lag2 = st.number_input("Units sold two months ago", 0.0, 500.0, float(m7["units"]), key=f"p_l2{pid}")
        ord1 = st.number_input("Orders containing it last month", 0.0, 100.0, float(m8["orders"]), key=f"p_o{pid}")
    with c2:
        price = st.number_input("Unit price", 0.0, 1000.0, float(m8["UnitPrice"]), key=f"p_pr{pid}")
        stock = st.number_input("Total stock across warehouses", 0.0, 20000.0, float(m8["total_stock"]), key=f"p_st{pid}")
    with c3:
        nwh = st.number_input("Warehouses stocking it", 0, 15, int(m8["n_warehouses"]), key=f"p_nw{pid}")
        st.write(f"Category: **{m8['Category']}**")
    row = pd.DataFrame([{"lag_1": lag1, "lag_2": lag2, "lag_mean": (lag1 + lag2) / 2,
                         "orders_lag_1": ord1, "UnitPrice": price, "total_stock": stock,
                         "n_warehouses": nwh, "Category": m8["Category"]}])
    if st.button("Forecast demand", key="b_dem"):
        pred = float(m["pipeline"].predict(row[m["features"]])[0])
        regression_output("demand", pred, "Predicted units next month", "{:.1f}")
        st.write(f"Baseline for comparison: the average product sold about {demand['units'].mean():.1f} units per month.")

# Customer value
with tab3:
    banner("customer")
    m = bundle["customer"]
    cid = st.selectbox("Customer", cust["CustomerID"].tolist(), key="c_id")
    r = cust[cust["CustomerID"] == cid].iloc[0]
    c1, c2, c3 = st.columns(3)
    with c1:
        n_ord = st.number_input("Completed orders (Jan to May)", 0, 50, int(r["n_orders"]), key=f"c_no{cid}")
        n_can = st.number_input("Cancelled orders", 0, 50, int(r["n_cancelled"]), key=f"c_nc{cid}")
        n_lines = st.number_input("Order lines", 0, 200, int(r["n_lines"]), key=f"c_nl{cid}")
        n_cat = st.number_input("Product categories bought", 0, 3, int(r["n_categories"]), key=f"c_ncat{cid}")
    with c2:
        qty = st.number_input("Total quantity", 0, 2000, int(r["total_qty"]), key=f"c_q{cid}")
        spend = st.number_input("Total spend", 0.0, 200000.0, float(r["total_spend"]), step=100.0, key=f"c_s{cid}")
        rec = st.number_input("Days since last order", 0, 151, int(r["recency_days"]), key=f"c_r{cid}")
        ten = st.number_input("Days since first order", 0, 151, int(r["tenure_days"]), key=f"c_t{cid}")
    with c3:
        se = st.slider("Share of spend: Electronics", 0.0, 1.0, float(r["share_Electronics"]), key=f"c_se{cid}")
        sf = st.slider("Share of spend: Fashion", 0.0, 1.0, float(r["share_Fashion"]), key=f"c_sf{cid}")
        sfo = st.slider("Share of spend: Food", 0.0, 1.0, float(r["share_Food"]), key=f"c_sfo{cid}")
        st.write(f"City: **{r['City']}**")
    tot = se + sf + sfo
    se, sf, sfo = (se / tot, sf / tot, sfo / tot) if tot > 0 else (0.0, 0.0, 0.0)
    n_all = n_ord + n_can
    row = pd.DataFrame([{
        "n_orders_all": n_all, "n_cancelled": n_can,
        "cancel_rate": n_can / n_all if n_all else 0.0, "n_orders": n_ord, "n_lines": n_lines,
        "total_qty": qty, "total_spend": spend, "n_categories": n_cat,
        "avg_order_value": spend / n_ord if n_ord else 0.0, "recency_days": rec,
        "tenure_days": ten, "share_Electronics": se, "share_Fashion": sf,
        "share_Food": sfo, "City": r["City"]}])
    if st.button("Predict future spend", key="b_cust"):
        pred = float(m["pipeline"].predict(row[m["features"]])[0])
        regression_output("customer", pred, "Predicted spend, Jun to Aug", "{:,.0f}")
        st.write(f"Baseline for comparison: the average customer spent about "
                 f"{cust['future_spend'].mean():,.0f} in that period.")

# Warehouse
with tab4:
    banner("warehouse")
    m = bundle["warehouse"]
    wid = st.selectbox("Warehouse (forecasting September 2026)", sorted(WH_CITY), key="w_id")
    last = wh[wh["WarehouseID"] == wid].sort_values("month").iloc[-1]
    c1, c2 = st.columns(2)
    with c1:
        lag_dr = st.slider("Delay rate last month", 0.0, 1.0, float(last["delay_rate"]), key=f"w_dr{wid}")
        lag_dd = st.number_input("Average delay days last month", 0.0, 7.0, float(last["avg_delay_days"]), key=f"w_dd{wid}")
        lag_ns = st.number_input("Shipments last month", 0, 200, int(last["n_shipments"]), key=f"w_ns{wid}")
        lag_av = st.number_input("Average order value last month", 0.0, 20000.0, float(last["avg_order_value"]), key=f"w_av{wid}")
    with c2:
        st.write("Warehouse profile (from the stock snapshot)")
        st.write(f"City: **{last['WarehouseCity']}**")
        st.write(f"Total stock: **{last['total_stock']:,.0f}**, products stocked: **{int(last['n_products'])}**")
        st.write(f"Average stock per product: **{last['avg_stock']:.0f}**, low-stock products: **{int(last['n_low_stock'])}**")
    row = pd.DataFrame([{
        "lag_delay_rate": lag_dr, "lag_avg_delay_days": lag_dd, "lag_n_shipments": lag_ns,
        "lag_avg_order_value": lag_av, "total_stock": last["total_stock"],
        "n_products": last["n_products"], "avg_stock": last["avg_stock"],
        "n_low_stock": last["n_low_stock"], "WarehouseCity": last["WarehouseCity"]}])
    if st.button("Predict delay rate", key="b_wh"):
        pred = float(m["pipeline"].predict(row[m["features"]])[0])
        regression_output("warehouse", pred, "Predicted delay rate next month", "{:.1%}")
        st.write("Baseline for comparison: the average warehouse-month had a delay rate of about 64%.")