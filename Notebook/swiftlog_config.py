FEATURES = {
    "delay": dict(
        target="is_delayed", task="classification",
        num=["n_lines", "total_qty", "order_value", "n_categories", "max_qty",
             "has_items", "order_month", "order_dow", "cross_city", "wh_load_7d"],
        cat=["WarehouseID", "WarehouseCity", "CustomerCity"]),
    "demand": dict(
        target="units", task="regression",
        num=["lag_1", "lag_2", "lag_mean", "orders_lag_1", "UnitPrice",
             "total_stock", "n_warehouses"],
        cat=["Category"]),
    "customer": dict(
        target="future_spend", task="regression",
        num=["n_orders_all", "n_cancelled", "cancel_rate", "n_orders", "n_lines",
             "total_qty", "total_spend", "n_categories", "avg_order_value",
             "recency_days", "tenure_days",
             "share_Electronics", "share_Fashion", "share_Food"],
        cat=["City"]),
    "warehouse": dict(
        target="delay_rate", task="regression",
        num=["lag_delay_rate", "lag_avg_delay_days", "lag_n_shipments",
             "lag_avg_order_value", "total_stock", "n_products",
             "avg_stock", "n_low_stock"],
        cat=["WarehouseCity"]),
}