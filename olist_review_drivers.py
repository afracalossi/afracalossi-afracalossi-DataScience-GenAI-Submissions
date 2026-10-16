"""
Olist (Brazilian e-commerce): which factors drive a 4-5 star review?

Target: y = 1 if review_score is 4 or 5; 0 if 1-3 or the order has no review.
Population: delivered orders (so shipping time exists).

Usage:  python olist_review_drivers.py <folder_with_olist_csvs> [--no-model]
Needs:  pandas numpy scipy scikit-learn statsmodels

Steps
  1. build_table   - join orders, customers, items, products, sellers, payments,
                     reviews and geolocation into one row per order.
  2. driver_ranking - one logistic regression with all factors; each factor is
                     ranked by the likelihood-ratio drop when removed, plus
                     average marginal effects (percentage points).
  3. predictive    - chronological 70/15/15 split; logistic regression vs
                     gradient boosting, AUC on the latest 15% of orders.
"""
import sys, warnings
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")


def build_table(D):
    rd = lambda f, **k: pd.read_csv(f"{D}/{f}", **k)
    orders = rd("olist_orders_dataset.csv", parse_dates=[3, 4, 5, 6, 7])
    cust = rd("olist_customers_dataset.csv", dtype={"customer_zip_code_prefix": str})
    items = rd("olist_order_items_dataset.csv")
    pay = rd("olist_order_payments_dataset.csv")
    prod = rd("olist_products_dataset.csv")
    trans = rd("product_category_name_translation.csv", encoding="utf-8-sig")
    sell = rd("olist_sellers_dataset.csv", dtype={"seller_zip_code_prefix": str})
    geo = rd("olist_geolocation_dataset.csv", dtype={"geolocation_zip_code_prefix": str})
    rev = rd("olist_order_reviews_dataset.csv", parse_dates=["review_answer_timestamp"],
             date_format="%d/%m/%Y %H:%M")

    # target: latest review per order
    rev = rev.sort_values("review_answer_timestamp").drop_duplicates("order_id", keep="last")
    df = orders.merge(rev[["order_id", "review_score"]], on="order_id", how="left")
    df["has_review"] = df.review_score.notna().astype(int)
    df["y"] = (df.review_score >= 4).astype(int)

    # geography via zip-prefix centroids
    g = geo.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]].mean()
    df = df.merge(cust, on="customer_id", how="left")
    df = df.merge(g.rename(columns={"geolocation_lat": "c_lat", "geolocation_lng": "c_lng"}),
                  left_on="customer_zip_code_prefix", right_index=True, how="left")
    df = df.sort_values("order_purchase_timestamp")
    df["prior_orders"] = df.groupby("customer_unique_id").cumcount()   # no look-ahead

    # items, products, sellers
    prod = prod.merge(trans, on="product_category_name", how="left")
    it = items.merge(prod[["product_id", "product_category_name_english"]], on="product_id", how="left")
    it = it.merge(sell, on="seller_id", how="left")
    it = it.merge(g.rename(columns={"geolocation_lat": "s_lat", "geolocation_lng": "s_lng"}),
                  left_on="seller_zip_code_prefix", right_index=True, how="left")
    agg = it.groupby("order_id").agg(
        n_items=("order_item_id", "count"), n_sellers=("seller_id", "nunique"),
        price_total=("price", "sum"), freight_total=("freight_value", "sum"),
        s_lat=("s_lat", "mean"), s_lng=("s_lng", "mean"), seller_state=("seller_state", "first"),
        category=("product_category_name_english", "first"))
    df = df.merge(agg, on="order_id", how="left")

    # payments (voucher is the only loyalty/discount proxy in this data)
    pg = pay.groupby("order_id").agg(installments=("payment_installments", "max"))
    v = pay[pay.payment_type == "voucher"].groupby("order_id").payment_value.sum().rename("voucher_value")
    df = df.merge(pg, on="order_id", how="left").merge(v, on="order_id", how="left")
    df["used_voucher"] = (df.voucher_value.fillna(0) > 0).astype(int)

    # derived
    def hav(a, b, c, d):
        a, b, c, d = map(np.radians, (a, b, c, d))
        h = np.sin((c - a) / 2) ** 2 + np.cos(a) * np.cos(c) * np.sin((d - b) / 2) ** 2
        return 6371 * 2 * np.arcsin(np.sqrt(h))
    days = lambda a, b: (df[a] - df[b]).dt.total_seconds() / 86400
    df["dist_km"] = hav(df.s_lat, df.s_lng, df.c_lat, df.c_lng)
    df["est_days"] = days("order_estimated_delivery_date", "order_purchase_timestamp")
    df["delay_days"] = days("order_delivered_customer_date", "order_estimated_delivery_date")  # >0 = late
    df["handoff_days"] = days("order_delivered_carrier_date", "order_purchase_timestamp")
    df["transit_days"] = days("order_delivered_customer_date", "order_delivered_carrier_date")
    df["ship_days"] = days("order_delivered_customer_date", "order_purchase_timestamp")
    df["is_late"] = (df.delay_days > 0).astype(int)
    df["freight_ratio"] = df.freight_total / (df.price_total + df.freight_total)

    m = df[(df.order_status == "delivered") & df.order_delivered_customer_date.notna()].copy()
    print(f"delivered orders: {len(m):,} | positive rate {m.y.mean():.3f} | no-review {1 - m.has_review.mean():.3f}")
    return m


def driver_ranking(m):
    m = m.dropna(subset=["delay_days", "handoff_days", "transit_days", "est_days", "price_total",
                         "freight_ratio", "dist_km", "installments", "category"]).copy()
    m["late_b"] = pd.cut(m.delay_days, [-1e3, -10, -3, 0, 3, 7, 1e3],
                         labels=["a_early10+", "b_early3-10", "c_0-3early", "d_late0-3", "e_late3-7", "f_late7+"]).astype(str)
    m["handoff_b"] = pd.cut(m.handoff_days, [-1, 1, 3, 5, 1e3], labels=["a_<=1d", "b_1-3d", "c_3-5d", "d_5d+"]).astype(str)
    m["transit_b"] = pd.qcut(m.transit_days, 5, labels=["t1_fastest", "t2", "t3", "t4", "t5_slowest"]).astype(str)
    m["promise_b"] = pd.qcut(m.est_days, 4, labels=["p1_short", "p2", "p3", "p4_long"]).astype(str)
    m["multi_seller"] = (m.n_sellers > 1).astype(int)
    m["items_b"] = m.n_items.clip(upper=3).astype(int).astype(str)
    m["price_q"] = pd.qcut(m.price_total, 5, labels=["q1_cheap", "q2", "q3", "q4", "q5_pricey"]).astype(str)
    m["freight_q"] = pd.qcut(m.freight_ratio, 5, labels=["f1_low", "f2", "f3", "f4", "f5_high"]).astype(str)
    m["dist_q"] = pd.qcut(m.dist_km, 5, labels=["d1_near", "d2", "d3", "d4", "d5_far"], duplicates="drop").astype(str)
    m["inst_b"] = pd.cut(m.installments, [-1, 1, 4, 8, 100], labels=["i1", "i2-4", "i5-8", "i9+"]).astype(str)
    m["repeat"] = (m.prior_orders > 0).astype(int)
    m["cat"] = m.category.where(m.category.isin(m.category.value_counts().index[:12]), "other")
    m["state_g"] = m.customer_state.where(m.customer_state.isin(["SP", "RJ", "MG", "BA", "RS", "PR"]), "other")

    groups = {
        "Lateness vs promised date": "C(late_b)", "Number of items": "C(items_b)",
        "Multi-seller order": "multi_seller", "Seller dispatch speed": "C(handoff_b)",
        "Product category": "C(cat)", "Carrier transit time": "C(transit_b)",
        "Delivery promise length": "C(promise_b)", "Customer state": "C(state_g)",
        "Instalments": "C(inst_b)", "Seller-customer distance": "C(dist_q)",
        "Voucher used": "used_voucher", "Freight as % of order": "C(freight_q)",
        "Repeat customer": "repeat", "Price": "C(price_q)"}
    full = " + ".join(groups.values())

    for lab, d in [("A: no review = 0", m), ("B: reviewed orders only", m[m.has_review == 1])]:
        f = smf.logit(f"y ~ {full}", d).fit(disp=0, maxiter=200)
        rows = []
        for g, _ in groups.items():
            rest = " + ".join(v for k, v in groups.items() if k != g)
            r = smf.logit(f"y ~ {rest}", d).fit(disp=0, maxiter=200)
            lr, k = 2 * (f.llf - r.llf), f.df_model - r.df_model
            rows.append((g, lr, int(k), stats.chi2.sf(lr, k)))
        t = pd.DataFrame(rows, columns=["factor", "LR_chi2", "df", "p"]).sort_values("LR_chi2", ascending=False)
        t["share_%"] = 100 * t.LR_chi2 / t.LR_chi2.sum()
        print(f"\n===== {lab} | n={len(d):,} | base rate {d.y.mean():.3f} | pseudo-R2 {f.prsquared:.3f}")
        print(t.round(4).to_string(index=False))
        ame = f.get_margeff(at="overall", method="dydx", dummy=True).summary_frame()
        ame["pp"] = 100 * ame["dy/dx"]
        ame = ame[~ame.index.str.contains("Intercept")]
        ame = ame.reindex(ame.pp.abs().sort_values(ascending=False).index)
        print("\nTop effects, percentage-point change in P(4-5 star) vs the reference level "
              "(reference = first level alphabetically: early 10+ days, dispatch <=1d, fastest transit, "
              "short promise, 1 item, cheapest, nearest):")
        print(ame.head(20)[["pp", "Pr(>|z|)"]].round(3).to_string())


def predictive(m):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    num = ["ship_days", "est_days", "delay_days", "handoff_days", "transit_days", "dist_km", "price_total",
           "freight_total", "freight_ratio", "n_items", "n_sellers", "installments", "used_voucher", "prior_orders"]
    cat = ["customer_state", "category", "seller_state"]
    m = m.sort_values("order_purchase_timestamp").reset_index(drop=True)
    for lab, d in [("A: no review = 0", m), ("B: reviewed only", m[m.has_review == 1].reset_index(drop=True))]:
        tr, te = d.iloc[:int(len(d) * .85)], d.iloc[int(len(d) * .85):]   # latest 15% held out
        lr = Pipeline([("p", ColumnTransformer([
            ("n", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num),
            ("c", OneHotEncoder(handle_unknown="ignore", min_frequency=50), cat)])),
            ("m", LogisticRegression(max_iter=2000, C=0.5))])
        gb = Pipeline([("p", ColumnTransformer([
            ("n", "passthrough", num),
            ("c", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1), cat)])),
            ("m", HistGradientBoostingClassifier(learning_rate=.05, max_iter=300, max_leaf_nodes=15,
                  min_samples_leaf=100, l2_regularization=1.0,
                  categorical_features=list(range(len(num), len(num) + len(cat))), random_state=0))])
        for name, mdl in [("logistic", lr), ("gradient boosting", gb)]:
            mdl.fit(tr[num + cat], tr.y)
            auc = roc_auc_score(te.y, mdl.predict_proba(te[num + cat])[:, 1])
            print(f"[{lab}] {name}: time-split test AUC = {auc:.3f} (n_test={len(te):,}, base rate {te.y.mean():.3f})")


if __name__ == "__main__":
    m = build_table(sys.argv[1])
    driver_ranking(m)
    if "--no-model" not in sys.argv:
        print("\n===== Predictive check (chronological split)")
        predictive(m)
