"""
retail_agent.py

Retail KPI engine:
. loads sales, product and competitor data
. validates input schema
. builds stable KPIs
. supports grouping by store or region
. provides insight tables for price, stock and markdown flags
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass


# ---------------------------------------------------------
# 1. Config
# ---------------------------------------------------------

@dataclass
class RetailAgentConfig:
    sales_path: str
    products_path: str
    competitors_path: str
    min_history_days: int = 30
    overstock_days_cover: float = 60.0
    understock_days_cover: float = 7.0
    high_discount_threshold: float = 0.30
    high_price_index_threshold: float = 1.20
    low_price_index_threshold: float = 0.80


# ---------------------------------------------------------
# 2. Validation helper
# ---------------------------------------------------------

def validate_columns(df, required, df_name):
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{df_name} missing required columns: {missing}")


# ---------------------------------------------------------
# 3. RetailInsightAgent
# ---------------------------------------------------------

class RetailInsightAgent:

    def __init__(self, config: RetailAgentConfig):
        self.config = config
        self.sales = None
        self.products = None
        self.competitors = None
        self.kpis = None

    # -----------------------------
    # Load and validate data
    # -----------------------------
    def load_data(self) -> None:
        self.sales = pd.read_csv(self.config.sales_path, parse_dates=["date"])
        self.products = pd.read_csv(self.config.products_path)
        self.competitors = pd.read_csv(self.config.competitors_path, parse_dates=["date"])

        validate_columns(
            self.sales,
            ["date", "sku", "units", "revenue", "stock_on_hand"],
            "sales"
        )

        validate_columns(
            self.products,
            ["sku"],
            "products"
        )

        validate_columns(
            self.competitors,
            ["date", "sku", "price"],
            "competitors"
        )

        for col in ["units", "revenue", "stock_on_hand"]:
            self.sales[col] = pd.to_numeric(self.sales[col], errors="coerce").fillna(0)

        self.competitors["price"] = pd.to_numeric(self.competitors["price"], errors="coerce").fillna(0)

        self.competitors = (
            self.competitors
            .sort_values(["sku", "date"])
            .groupby("sku")
            .apply(lambda g: g.ffill())
            .reset_index(drop=True)
        )

    # -----------------------------
    # KPI builder
    # -----------------------------
    def build_kpis(self) -> None:
        df = self.sales.merge(self.products, on="sku", how="left")

        df["unit_price"] = np.where(df["units"] > 0, df["revenue"] / df["units"], np.nan)

        daily = (
            df.groupby(["date", "sku"], as_index=False)
              .agg(
                  units_sold=("units", "sum"),
                  revenue=("revenue", "sum"),
                  stock_on_hand=("stock_on_hand", "max"),
                  unit_price=("unit_price", "mean")
              )
        )

        max_date = daily["date"].max()
        min_required = max_date - pd.Timedelta(days=self.config.min_history_days)
        earliest_date = daily["date"].min()
        min_date = earliest_date if earliest_date > min_required else min_required

        recent = daily[daily["date"].between(min_date, max_date)]

        sku_stats = (
            recent.groupby("sku", as_index=False)
            .agg(
                days_sold=("units_sold", lambda x: (x > 0).count()),
                total_units=("units_sold", "sum"),
                total_revenue=("revenue", "sum"),
                avg_units_per_day=("units_sold", "mean"),
                avg_price=("unit_price", "mean"),
                avg_stock_on_hand=("stock_on_hand", "mean"),
            )
        )

        sku_stats["sell_through_rate"] = (
            sku_stats["avg_units_per_day"] 
            / sku_stats["avg_stock_on_hand"].replace(0, np.nan)
        )

        sku_stats["days_of_cover"] = (
            sku_stats["avg_stock_on_hand"] 
            / sku_stats["avg_units_per_day"].replace(0, np.nan)
        ).clip(lower=0, upper=365)

        comp_daily = (
            self.competitors.groupby(["date", "sku"], as_index=False)
            .agg(
                competitor_min_price=("price", "min"),
                competitor_avg_price=("price", "mean")
            )
        )

        recent_comp = comp_daily[comp_daily["date"].between(min_date, max_date)]

        comp_stats = (
            recent_comp.groupby("sku", as_index=False)
            .agg(
                competitor_min_price=("competitor_min_price", "mean"),
                competitor_avg_price=("competitor_avg_price", "mean")
            )
        )

        sku_stats = sku_stats.merge(comp_stats, on="sku", how="left")

        sku_stats["price_index_vs_avg"] = (
            sku_stats["avg_price"] 
            / sku_stats["competitor_avg_price"].replace(0, np.nan)
        )

        if "full_price" in self.products.columns:
            price_cols = ["sku", "full_price"]
            if "cost_price" in self.products.columns:
                price_cols.append("cost_price")

            sku_stats = sku_stats.merge(self.products[price_cols], on="sku", how="left")

            sku_stats["avg_discount"] = (
                (sku_stats["full_price"] - sku_stats["avg_price"])
                / sku_stats["full_price"].replace(0, np.nan)
            ).fillna(0)
        else:
            sku_stats["avg_discount"] = 0.0

        sku_stats["flag_overstock"] = sku_stats["days_of_cover"] > self.config.overstock_days_cover
        sku_stats["flag_understock"] = sku_stats["days_of_cover"] < self.config.understock_days_cover
        sku_stats["flag_heavy_discount"] = sku_stats["avg_discount"] >= self.config.high_discount_threshold
        sku_stats["flag_overpriced_vs_market"] = sku_stats["price_index_vs_avg"] >= self.config.high_price_index_threshold
        sku_stats["flag_underpriced_vs_market"] = sku_stats["price_index_vs_avg"] <= self.config.low_price_index_threshold

        if "cost_price" in sku_stats.columns:
            sku_stats["gross_margin_pct"] = (
                (sku_stats["avg_price"] - sku_stats["cost_price"])
                / sku_stats["avg_price"].replace(0, np.nan)
            )
        else:
            sku_stats["gross_margin_pct"] = np.nan

        sku_stats["unit_velocity"] = sku_stats["total_units"] / self.config.min_history_days
        sku_stats["revenue_velocity"] = sku_stats["total_revenue"] / self.config.min_history_days

        total_rev = sku_stats["total_revenue"].sum()
        sku_stats["revenue_contribution_pct"] = (
            sku_stats["total_revenue"] / total_rev if total_rev > 0 else 0
        )

        self.kpis = sku_stats

    # -----------------------------
    # Insight helper
    # -----------------------------
    def find(self, mask, sort_by, cols, ascending=False, n=20):
        df = self.kpis.copy()
        df = df[mask(df)]
        return df.sort_values(sort_by, ascending=ascending)[cols].head(n)

    # -----------------------------
    # Insights
    # -----------------------------
    def top_overstock_risks(self, n=20):
        return self.find(
            mask=lambda d: (d["flag_overstock"]) & (d["total_units"] >= 0),
            sort_by="days_of_cover",
            cols=[
                "sku", "days_of_cover", "avg_stock_on_hand",
                "avg_units_per_day", "total_units", "total_revenue"
            ],
            ascending=False,
            n=n
        )

    def top_underpriced_winners(self, n=20):
        return self.find(
            mask=lambda d: (
                (d["avg_units_per_day"] > 0)
                & (d["price_index_vs_avg"].notna())
                & (d["price_index_vs_avg"] < 1.0)
            ),
            sort_by="avg_units_per_day",
            cols=["sku", "avg_units_per_day", "price_index_vs_avg", "avg_price", "competitor_avg_price"],
            ascending=False,
            n=n
        )

    def heavy_discount_skus(self, n=20):
        return self.find(
            mask=lambda d: d["flag_heavy_discount"],
            sort_by="avg_discount",
            cols=["sku", "avg_discount", "total_units", "total_revenue", "days_of_cover"],
            ascending=False,
            n=n
        )

    # -----------------------------
    # JSON export
    # -----------------------------
    def to_json(self):
        return self.kpis.to_dict(orient="records") if self.kpis is not None else []
