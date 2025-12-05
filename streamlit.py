import streamlit as st
import pandas as pd
import requests
from retail_agent import RetailInsightAgent, RetailAgentConfig


# ---------------------------------------------------------
# TinyDolphin LLM Commentary (correct API)
# ---------------------------------------------------------
def generate_llm_commentary(df, topic):
    if df is None or df.empty:
        return f"No {topic} signals detected."

    rows = df.to_dict(orient="records")

    # Build prompt for generate API
    prompt = f"""
Act as a senior retail merchandiser.
Topic: {topic}
Data rows: {rows}

Write clear, direct, practical insights. 
Avoid fluff.
"""

    try:
        # Correct API for TinyDolphin
        response = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={
                "model": "tinydolphin",
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )

        data = response.json()

        # TinyDolphin returns text inside "response"
        if "response" in data and data["response"]:
            return data["response"].strip()

        return "Model returned no usable content."

    except Exception as e:
        return f"Ollama error: {str(e)}"


# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
st.set_page_config(page_title="Retail Insight Console", layout="wide")
st.title("Retail Insight Console")

st.sidebar.header("Upload data")
sales_file = st.sidebar.file_uploader("Sales CSV", type="csv")
products_file = st.sidebar.file_uploader("Products CSV", type="csv")
competitors_file = st.sidebar.file_uploader("Competitor Prices CSV", type="csv")

group_choice = st.sidebar.selectbox("Group by", ["none", "store", "region"])
top_n = st.sidebar.slider("Top N results", 5, 50, 20)

run_button = st.sidebar.button("Generate Insights")


# ---------------------------------------------------------
# Workflow
# ---------------------------------------------------------
if run_button:
    if not sales_file or not products_file or not competitors_file:
        st.error("Upload all three files before running.")
        st.stop()

    sales_df = pd.read_csv(sales_file, parse_dates=["date"])
    products_df = pd.read_csv(products_file)
    competitors_df = pd.read_csv(competitors_file, parse_dates=["date"])

    # Grouping logic
    if group_choice in ["store", "region"] and group_choice in sales_df.columns:
        sales_df["group_key"] = sales_df[group_choice]
    else:
        sales_df["group_key"] = "all"

    groups = sales_df["group_key"].unique()
    st.subheader("Insights by Group")

    for grp in groups:
        st.markdown(f"## Group: {grp}")

        group_sales = sales_df[sales_df["group_key"] == grp]

        # Save temp CSVs for RetailAgent
        group_sales.to_csv("temp_sales.csv", index=False)
        products_df.to_csv("temp_products.csv", index=False)
        competitors_df.to_csv("temp_competitors.csv", index=False)

        config = RetailAgentConfig(
            sales_path="temp_sales.csv",
            products_path="temp_products.csv",
            competitors_path="temp_competitors.csv"
        )

        agent = RetailInsightAgent(config)
        agent.load_data()
        agent.build_kpis()

        with st.expander("View KPI Table"):
            st.dataframe(agent.kpis)

        # Insight tables
        overstock = agent.top_overstock_risks(n=top_n)
        underpriced = agent.top_underpriced_winners(n=top_n)
        markdown = agent.heavy_discount_skus(n=top_n)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("Overstock Risks")
            st.dataframe(overstock)

        with col2:
            st.write("Underpriced Winners")
            st.dataframe(underpriced)

        with col3:
            st.write("Markdown Candidates")
            st.dataframe(markdown)

        # -------------------------------------------------
        # TinyDolphin Commentary
        # -------------------------------------------------
        st.markdown("### AI Commentary (TinyDolphin)")

        st.markdown("#### Price Actions")
        st.write(generate_llm_commentary(underpriced, "price actions"))

        st.markdown("#### Stock Actions")
        st.write(generate_llm_commentary(overstock, "stock actions"))

        st.markdown("#### Markdown Actions")
        st.write(generate_llm_commentary(markdown, "markdown strategy"))

else:
    st.info("Upload all CSVs and press Generate Insights.")
