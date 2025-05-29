import streamlit as st
import pandas as pd
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import os

# Need to import the Base and table models from green_anole
# Assuming green_anole.py is in the same directory or adjust path accordingly
try:
    from src.fast.green_anole import Base, EnrichedData, Feedback, ENGINE
except ImportError as e:
    st.error(f"Could not import database models from src.fast.green_anole. Error: {e}")
    st.stop()

# Create session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=ENGINE)

st.title("📊 Database Table Viewer")


# --- Helper function to load data ---
@st.cache_data(ttl=60)  # Cache for 1 minute
def load_table_data(table_model):
    """Loads data from the specified SQLAlchemy model."""
    db = SessionLocal()
    try:
        query = db.query(table_model)
        df = pd.read_sql(query.statement, db.bind)

        # --- Flatten ai_evaluation if it's the enriched_data table ---
        if table_model == EnrichedData and "ai_evaluation" in df.columns:
            # Define the keys we want to extract
            score_keys = ["SEP", "DCI", "CCC", "Communication", "Overall"]
            feedback_keys = ["strengths", "suggestions"]
            other_keys = ["annotated_response"]

            # Function to safely extract nested values
            def safe_get(data, keys, default=None):
                try:
                    for key in keys:
                        data = data[key]
                    return data
                except (TypeError, KeyError, IndexError):
                    return default

            # Create new columns
            for key in other_keys:
                df[f"ai_{key}"] = df["ai_evaluation"].apply(
                    lambda x: safe_get(x, [key])
                )
            for key in score_keys:
                df[f"ai_score_{key}"] = df["ai_evaluation"].apply(
                    lambda x: safe_get(x, ["rubric_scores", key])
                )
            for key in feedback_keys:
                df[f"ai_feedback_{key}"] = df["ai_evaluation"].apply(
                    lambda x: safe_get(x, ["feedback", key])
                )

            # Drop the original complex column
            df = df.drop(columns=["ai_evaluation"])
        # --- End Flattening ---
        else:
            # For other tables or if ai_evaluation is missing, convert objects/dicts to strings
            for col in df.columns:
                if df[col].dtype == "object":
                    try:
                        df[col] = df[col].apply(
                            lambda x: str(x) if isinstance(x, (dict, list)) else x
                        )
                    except Exception:
                        pass  # Ignore errors

        return df
    except Exception as e:
        st.error(f"Error loading data for {table_model.__tablename__}: {e}")
        return pd.DataFrame()
    finally:
        db.close()


# --- Table Selection ---
# Get table names and model classes from the Base registry
available_tables = {
    mapper.class_.__tablename__: mapper.class_ for mapper in Base.registry.mappers
}

selected_table_name = st.selectbox(
    "Select Table to View:", options=list(available_tables.keys()), index=0
)

if selected_table_name:
    selected_model = available_tables[selected_table_name]
    st.subheader(f"Viewing Table: `{selected_table_name}`")

    df_table = load_table_data(selected_model)

    if not df_table.empty:
        st.dataframe(df_table, use_container_width=True)

        # --- Export Button ---
        csv = df_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"💾 Download {selected_table_name}.csv",
            data=csv,
            file_name=f"{selected_table_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key=f"download_{selected_table_name}",  # Unique key for download button
        )
    else:
        st.warning("No data found in this table or an error occurred.")
else:
    st.info("Please select a table to view.")

st.caption("Data refreshes every 60 seconds.")
