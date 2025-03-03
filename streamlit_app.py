# streamlit_app.py

import streamlit as st
from google.oauth2 import service_account
from google.cloud import bigquery
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import numpy as np
from scipy.stats import beta

# Create API client.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = bigquery.Client(credentials=credentials)

# Perform query.
@st.cache_data(ttl=600)
def run_query(query):
    try:
        query_job = client.query(query)
        rows_raw = query_job.result()
        rows = [dict(row) for row in rows_raw]
        df = pd.DataFrame(rows)
        return df
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

def calculate_conversion_rates(df, event1_col, event2_col, assignment_col):
    if df.empty or event1_col not in df.columns or event2_col not in df.columns or assignment_col not in df.columns:
        return None, None
    df["converted"] = ~df[event2_col].isnull()
    conversion_rates = (
        df.groupby(assignment_col)["converted"].agg(["count", "sum"]).reset_index()
    )
    conversion_rates["conversion_rate"] = (
        conversion_rates["sum"] / conversion_rates["count"]
    )
    if len(conversion_rates[assignment_col].unique()) == 2:
        if "A" in conversion_rates[assignment_col].unique() and "B" in conversion_rates[assignment_col].unique():
            variant_a_rate = conversion_rates[conversion_rates[assignment_col] == "A"]["conversion_rate"].iloc[0]
            variant_b_rate = conversion_rates[conversion_rates[assignment_col] == "B"]["conversion_rate"].iloc[0]
            lift_drop = (variant_b_rate - variant_a_rate) / variant_a_rate
        else:
            lift_drop = None
    else:
        lift_drop = None
    
    return conversion_rates, lift_drop

def create_horizontal_conversion_chart(conversion_rates, assignment_col):
    if conversion_rates is None:
        return None
    
    fig, ax = plt.subplots()
    bars = ax.barh(
        conversion_rates[assignment_col],
        conversion_rates["conversion_rate"],
    )
    
    ax.set_xlabel("Conversion Rate")
    ax.set_title("Conversion Rate by Assignment")
    ax.set_xlim(0,1)
    
    for bar in bars:
        width = bar.get_width()
        ax.annotate(f'{width:.2%}',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(-30, 0),
                    textcoords="offset points",
                    ha='right', va='center', fontweight='bold', fontsize=14)
    
    return fig

def perform_chi_squared_test(df, assignment_col):
    if df is None or assignment_col not in df.columns or "converted" not in df.columns:
        return None
    contingency_table = pd.crosstab(df[assignment_col], df["converted"])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    return chi2, p, dof, expected

def create_posterior_distribution_chart(df, assignment_col):
    if df is None or assignment_col not in df.columns or "converted" not in df.columns:
        return None

    variants = df[assignment_col].unique()
    if len(variants) != 2 or "A" not in variants or "B" not in variants:
      return None
    
    fig, ax = plt.subplots()
    x = np.linspace(0, 1, 1000)

    for variant in variants:
        variant_data = df[df[assignment_col] == variant]
        successes = variant_data["converted"].sum()
        failures = len(variant_data) - successes
        
        # Beta distribution parameters
        a = successes + 1  # Adding 1 for a weak prior (uniform distribution)
        b = failures + 1
        
        posterior = beta.pdf(x, a, b)
        ax.plot(x, posterior, label=f"{variant} (a={a}, b={b})")

    ax.set_xlabel("Conversion Rate")
    ax.set_ylabel("Posterior Density")
    ax.set_title("Posterior Distribution of Conversion Rates")
    ax.legend()
    return fig

st.title("BigQuery Query Interface")

# Initialize session state variables if they don't exist
if 'df' not in st.session_state:
    st.session_state.df = None
if "event1_column" not in st.session_state:
    st.session_state.event1_column = None
if "event2_column" not in st.session_state:
    st.session_state.event2_column = None
if "assignment_column" not in st.session_state:
    st.session_state.assignment_column = None

# Sidebar setup is moved outside the button logic
with st.sidebar:
    st.header("Column Selection")
    
    if st.session_state.df is not None:
        all_columns = st.session_state.df.columns.tolist()
    else:
        all_columns = [] #provide an empty list so the drop downs still render.

    st.session_state.event1_column = st.selectbox(
        "Select Event 1 Column:", all_columns, key="event1_column_selectbox", index= 0 if len(all_columns) > 0 else 0, disabled= len(all_columns) == 0
    )
    st.session_state.event2_column = st.selectbox(
        "Select Event 2 Column:", all_columns, key="event2_column_selectbox", index = 0 if len(all_columns) > 0 else 0, disabled= len(all_columns) == 0
    )
    st.session_state.assignment_column = st.selectbox(
        "Select Assignment Column:", all_columns, key="assignment_column_selectbox", index = 0 if len(all_columns) > 0 else 0, disabled= len(all_columns) == 0
    )
    

query_text = st.text_area("Enter your BigQuery SQL query here:", height=200)

if st.button("Run Query"):
    if query_text:
        with st.spinner("Running query..."):
            st.session_state.df = run_query(query_text)
        if st.session_state.df is not None:
            st.write("Sample Data:")
            st.table(st.session_state.df.head(10))
            
            
            conversion_rates, lift_drop = calculate_conversion_rates(
                st.session_state.df,
                st.session_state.event1_column,
                st.session_state.event2_column,
                st.session_state.assignment_column,
            )
            if conversion_rates is not None:
                st.session_state.df["converted"] = ~st.session_state.df[st.session_state.event2_column].isnull()

                conversion_chart = create_horizontal_conversion_chart(conversion_rates,st.session_state.assignment_column)
                if conversion_chart is not None:
                    st.pyplot(conversion_chart)
                if lift_drop is not None:
                    st.write(f"Lift/Drop of Variant B compared to Variant A: {lift_drop:.2%}")
                else:
                    st.write("Could not calculate lift/drop, ensure you have a proper assignment column with values A and B")
            else:
                st.write("There was a problem with one of your column selections, or there is not both A and B assignments in your data")

            chi2_result = perform_chi_squared_test(st.session_state.df, st.session_state.assignment_column)
            if chi2_result is not None:
              chi2, p, dof, expected = chi2_result
              st.write(f"Chi-Squared Test Results:")
              st.write(f"Chi-Squared Statistic: {chi2:.4f}")
              st.write(f"P-value: {p:.4f}")
              st.write(f"Degrees of Freedom: {dof}")

              alpha = 0.05
              if p < alpha:
                  st.write(f"Since the p-value ({p:.4f}) is less than alpha ({alpha}), we reject the null hypothesis. There is a statistically significant difference in conversion rates between the variants.")
              else:
                  st.write(f"Since the p-value ({p:.4f}) is greater than alpha ({alpha}), we fail to reject the null hypothesis. There is no statistically significant difference in conversion rates between the variants.")
            else:
              st.write("There was a problem with your column selections, could not perform chi-squared test")

            # Posterior distribution chart
            posterior_chart = create_posterior_distribution_chart(st.session_state.df, st.session_state.assignment_column)
            if posterior_chart is not None:
                st.pyplot(posterior_chart)
            else:
                st.write("Could not create posterior distribution chart. Ensure there are exactly two assignments named A and B")
    else:
        st.warning("Please enter a query.")
