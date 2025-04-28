import streamlit as st
import gspread
import pandas as pd
from google.oauth2.service_account import Credentials
import os
from dotenv import load_dotenv
import logging
import json
import time
from datetime import datetime

# --- SQLAlchemy Imports ---
from sqlalchemy import create_engine, Column, String, JSON, Boolean, DateTime, Integer
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError

# --- Langchain Imports ---
from langchain_openai import AzureChatOpenAI
from langchain import hub

# --- Local Imports ---


# Load environment variables first
load_dotenv()


# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger("streamlit")
# Ensure handler is present and set to configured level
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(getattr(logging, log_level))
    logger.addHandler(handler)
logger.setLevel(getattr(logging, log_level))
logger.debug(f"Log level set to: {log_level}")

# --- Constants ---
# Define column names from Google Sheet to avoid relying on index
# (These might need adjustment if the Sheet structure changes)
PART1_COL = "1. What do you notice about the behavior and body structures of the green lizards who live on islands with brown lizards versus the green lizards who live on islands with no brown lizards?"
PART2_COL = "2. Construct an explanation: How does natural selection explain the changes in the green lizard population over time?\n- Take out your Copy of Our General  Model\n- Use the key parts of the General Model to explain the cause and effect of the changes in the green lizard population over time.\nIn your explanation make sure to include the following:\n  - the key parts of the model\n  - cause and effect to describe how the population changed over time\n - the data about the green and brown lizard populations that support your explanation"

# Define the scope
SCOPE = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# Use SQLite database instead of CSV
# Construct absolute path relative to this script file
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, "enriched_data.db")
ENGINE = create_engine(f"sqlite:///{DB_PATH}")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=ENGINE)
Base = declarative_base()


# Define the SQLAlchemy model for enriched data
class EnrichedData(Base):
    __tablename__ = "enriched_data"

    # Using Timestamp (likely email) as the primary key
    Timestamp = Column(String, primary_key=True, index=True)
    ai_evaluation = Column(JSON)  # Store the JSON evaluation directly
    teacher_notes = Column(String, default="")
    teacher_score = Column(Integer, nullable=True)
    finalized = Column(Boolean, default=False)
    last_modified = Column(DateTime, default=datetime.now)


# Create the table if it doesn't exist
Base.metadata.create_all(bind=ENGINE)

# --- Load Langchain Hub Prompt ---
try:
    # Attempt to pull the prompt
    evaluation_prompt = hub.pull("hey-aw/openscied-rubric-b6-green-anole")
    logger.info("Successfully pulled Langchain Hub prompt.")
except Exception as e:
    logger.error(f"Failed to pull Langchain Hub prompt: {e}")
    # Fallback or error handling if the prompt cannot be pulled
    evaluation_prompt = None
    st.error(
        """
    ⚠️ Could not load the evaluation prompt from Langchain Hub.
    
    This means the AI evaluation features will not be available. Please check:
    1. Your internet connection
    2. Your Langchain Hub credentials
    3. The prompt URL (hey-aw/openscied-rubric-b6-green-anole)
    """
    )


def get_google_sheet():
    """Connect to Google Sheets and return the worksheet data"""
    try:
        # Debug: Print credentials path
        logger.debug("Attempting to load credentials...")
        creds_path = ".credentials/credentials.json"
        logger.debug(f"Looking for credentials at: {os.path.abspath(creds_path)}")

        # Load credentials from the .credentials directory
        credentials = Credentials.from_service_account_file(creds_path, scopes=SCOPE)
        logger.debug("Credentials loaded successfully")

        # Create a client to interact with Google Sheets
        client = gspread.authorize(credentials)
        logger.debug("Google Sheets client authorized")

        # Get the spreadsheet - using environment variable for sheet URL/ID
        sheet_url = os.getenv(
            "SHEET_URL",
            "https://docs.google.com/spreadsheets/d/1v0R6Rj89lCLr9xmWI993OJCL4ZszLS3wopJ8i6j2xoQ/edit?resourcekey=&gid=549865638#gid=549865638",
        )
        logger.debug(f"Using sheet URL: {sheet_url}")

        spreadsheet = client.open_by_url(sheet_url)
        logger.debug("Successfully opened spreadsheet")

        # Get the first worksheet
        worksheet = spreadsheet.get_worksheet(0)
        logger.debug("Accessed first worksheet")

        # Get all records
        data = worksheet.get_all_records()
        logger.debug(f"Retrieved {len(data)} records")

        # Debug raw data structure
        if data:
            logger.debug("First row data structure:")
            for key, value in data[0].items():
                logger.debug(f"  - {key}: {type(value)}")

        # Convert to DataFrame and show detailed column info
        df = pd.DataFrame(data)

        # Debug DataFrame structure
        logger.info("=== DataFrame Column Information ===")
        logger.info(f"All columns: {df.columns.tolist()}")
        logger.info("Column details:")
        for col in df.columns:
            logger.info(f"  - {col} (type: {df[col].dtype})")
            # Show first non-null value for each column
            first_val = df[col].dropna().iloc[0] if not df[col].isna().all() else None
            logger.info(f"    First value: {first_val}")
        logger.info("=================================")

        return df

    except Exception as e:
        logger.error(f"Error accessing Google Sheet: {str(e)}")
        logger.exception("Full traceback:")
        return None


def load_enriched_data():
    """Load enriched data from SQLite database"""
    db = SessionLocal()
    try:
        query = db.query(EnrichedData)
        df = pd.read_sql(query.statement, db.bind)
        logger.info(f"Successfully loaded {len(df)} records from {DB_PATH}")
        # ai_evaluation is already JSON, no need to parse again
        # Ensure correct dtypes after loading from DB
        if not df.empty:
            df["finalized"] = df["finalized"].astype(bool)
            df["last_modified"] = pd.to_datetime(df["last_modified"])
        else:
            # Ensure empty DataFrame has correct columns if DB was empty
            df = pd.DataFrame(
                columns=[
                    "Timestamp",
                    "ai_evaluation",
                    "teacher_notes",
                    "finalized",
                    "last_modified",
                ]
            )
        return df
    except SQLAlchemyError as e:
        logger.error(f"Error loading data from database {DB_PATH}: {e}")
        # Return empty DataFrame with correct columns on error
        return pd.DataFrame(
            columns=[
                "Timestamp",
                "ai_evaluation",
                "teacher_notes",
                "finalized",
                "last_modified",
            ]
        )
    finally:
        db.close()


def validate_enriched_data(df):
    # This function might be simplified or removed as type validation
    # is largely handled by SQLAlchemy and the database schema.
    # Keeping it for now to check column presence.
    """Validate the structure of the enriched data DataFrame"""
    required_columns = [
        "Timestamp",
        "ai_evaluation",
        "teacher_notes",
        "finalized",
        "last_modified",
    ]

    # Check required columns
    if not all(col in df.columns for col in required_columns):
        logger.error("Missing required columns in DataFrame")
        return False

    # Basic type check (optional, SQLAlchemy handles most)
    if not df.empty:
        if not pd.api.types.is_string_dtype(df["Timestamp"]):
            logger.error("Timestamp column is not string type")
            return False
        if not pd.api.types.is_string_dtype(df["teacher_notes"]):
            # Allow None/NaN before converting, check if it's string-like or None
            if (
                not df["teacher_notes"]
                .apply(lambda x: isinstance(x, str) or pd.isna(x))
                .all()
            ):
                logger.error("teacher_notes column has non-string values")
                return False
        if not pd.api.types.is_bool_dtype(df["finalized"]):
            logger.error("finalized column is not boolean type")
            return False
        if not pd.api.types.is_datetime64_any_dtype(df["last_modified"]):
            logger.error("last_modified column is not datetime type")
            return False
        # ai_evaluation is JSON, harder to validate type comprehensively here

    return True


def save_enriched_data(
    email, ai_output, teacher_notes="", teacher_score=None, finalized=False
):
    """Save or update enriched data for a response in the database."""
    db = SessionLocal()
    try:
        logger.info(f"Attempting to save enriched data for {email} to database")

        # Validate inputs
        if not email:
            logger.error("Email (Timestamp) is required")
            return False

        # Validate and convert AI output to JSON object (dict)
        ai_eval_json = None
        try:
            if isinstance(ai_output, str):
                # Handle empty strings or invalid JSON before parsing
                if ai_output.strip():
                    ai_eval_json = json.loads(ai_output)
                else:
                    ai_eval_json = None  # Or {} depending on desired default
            elif isinstance(ai_output, dict):
                ai_eval_json = ai_output  # Already a dict
            elif ai_output is None or pd.isna(ai_output):
                ai_eval_json = None  # Or {}
            else:
                logger.error(f"Invalid AI output type: {type(ai_output)}")
                return False
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in AI output: {e}")
            return False

        # Validate teacher_score
        score_to_save = None
        if teacher_score is not None and pd.notna(teacher_score):
            try:
                score_to_save = int(float(teacher_score))
            except (ValueError, TypeError):
                logger.error(
                    f"Invalid teacher_score value: {teacher_score}. Must be number-like."
                )
                return False

        # Check if record exists
        existing_record = (
            db.query(EnrichedData).filter(EnrichedData.Timestamp == email).first()
        )

        current_time = datetime.now()

        if existing_record:
            # Update existing record
            existing_record.ai_evaluation = ai_eval_json
            existing_record.teacher_notes = str(teacher_notes) if teacher_notes else ""
            existing_record.teacher_score = score_to_save
            existing_record.finalized = bool(finalized)
            existing_record.last_modified = current_time
            logger.debug(f"Updating existing record for {email}")
        else:
            # Create new record
            new_record = EnrichedData(
                Timestamp=email,
                ai_evaluation=ai_eval_json,
                teacher_notes=str(teacher_notes) if teacher_notes else "",
                teacher_score=score_to_save,
                finalized=bool(finalized),
                last_modified=current_time,
            )
            db.add(new_record)
            logger.debug(f"Creating new record for {email}")

        db.commit()
        logger.info(f"Successfully saved enriched data for {email} to database")
        return True

    except SQLAlchemyError as e:
        logger.error(f"Database error saving data for {email}: {e}")
        db.rollback()
        return False
    except Exception as e:
        # Catch other potential errors (e.g., during input validation)
        logger.error(f"Unexpected error saving data for {email}: {e}")
        db.rollback()
        return False
    finally:
        db.close()


def get_merged_data(raw_df):
    """Merge raw Google Forms data with enriched data"""
    try:
        # Load enriched data
        enriched_df = load_enriched_data()
        logger.info(f"Loaded enriched data with {len(enriched_df)} rows")

        # Debug info about enriched data
        logger.debug("Enriched data columns:")
        for col in enriched_df.columns:
            logger.debug(f"  - {col} (type: {enriched_df[col].dtype})")
            if len(enriched_df) > 0:
                sample = enriched_df[col].iloc[0]
                logger.debug(f"    Sample value: {sample} (type: {type(sample)})")

        # Merge the dataframes - **Corrected Merge Keys**
        merged_df = pd.merge(
            raw_df,
            enriched_df,
            # Use email from raw data and the key (which is email) from enriched data
            left_on="Email Address",
            right_on="Timestamp",  # This column in enriched_df holds the email
            how="left",
            suffixes=(
                "",
                "_enriched",
            ),  # Keep suffixes to identify columns from right df
        )

        # Clean up after merge: Drop the redundant timestamp column from enriched data
        if "Timestamp_enriched" in merged_df.columns:
            merged_df = merged_df.drop(columns=["Timestamp_enriched"])
        # We might also want to rename the original Timestamp from raw_df if needed
        # merged_df = merged_df.rename(columns={"Timestamp": "SubmissionTimestamp"})

        # Fill NaN values appropriately for columns from enriched_df
        # Ensure columns exist before trying to fillna
        for col in [
            "ai_evaluation",
            "teacher_notes",
            "finalized",
            "last_modified",
            "teacher_score",
        ]:
            if col not in merged_df.columns:
                merged_df[col] = pd.NA  # Or appropriate default

        merged_df["ai_evaluation"] = merged_df["ai_evaluation"].fillna({})
        merged_df["teacher_notes"] = merged_df["teacher_notes"].fillna("")
        # merged_df["finalized"] = merged_df["finalized"].fillna(False)
        # Ensure 'finalized' is boolean, handling potential NaNs from left join before fillna
        merged_df["finalized"] = merged_df["finalized"].fillna(False).astype(bool)
        merged_df["last_modified"] = pd.to_datetime(
            merged_df["last_modified"].fillna(pd.NaT)
        )
        # Ensure teacher_score is handled (assuming it might be numeric or None)
        # Convert to a numeric type that supports NaN, like float, if necessary, or handle appropriately
        # Example: Ensure it's float or object to allow Nones/NaNs
        if "teacher_score" in merged_df.columns:
            merged_df["teacher_score"] = pd.to_numeric(
                merged_df["teacher_score"], errors="coerce"
            )

        logger.info(f"Successfully merged data with {len(merged_df)} rows")
        # --- Add Debugging ---
        if logger.getEffectiveLevel() <= logging.DEBUG and not merged_df.empty:
            logger.debug("Merged DataFrame Head:\n" + merged_df.head().to_string())
            logger.debug("Sample ai_evaluation types in merged_df:")
            for i, val in enumerate(merged_df["ai_evaluation"].head()):
                logger.debug(f"  Row {i}: Type={type(val)}, Value={val}")
        # --- End Debugging ---
        return merged_df

    except Exception as e:
        logger.error(f"Error in get_merged_data: {e}")
        logger.exception("Full traceback:")
        return raw_df  # Return original data on error


def run_evaluation(llm, evaluation_prompt, part1, part2):
    """Run the evaluation for a student response."""
    # Combine Part 1 and Part 2 for the full response, handling optional Part 1
    if pd.isna(part1) or not part1.strip():
        full_response = part2
    else:
        full_response = f"Part 1:\n{part1}\n\nPart 2:\n{part2}"

    # Perform evaluation
    schema = evaluation_prompt.schema_
    structured_llm = llm.with_structured_output(schema)
    prompt_value = evaluation_prompt.format_prompt(response=full_response)
    input_val = (
        prompt_value.to_messages()
        if hasattr(prompt_value, "to_messages")
        else prompt_value
    )
    structured_output = structured_llm.invoke(input_val)
    output_data = (
        structured_output.dict()
        if hasattr(structured_output, "dict")
        else structured_output
    )

    return output_data


def batch_evaluate(df, llm, evaluation_prompt, batch_size=5):
    """Simple batch evaluation using LLM's built-in batching."""
    try:
        # Get unevaluated responses based on ai_evaluation being null/empty in the DataFrame
        # Assuming Timestamp is the unique key like email
        timestamp_col = "Timestamp"
        unevaluated_indices = df[
            df["ai_evaluation"].apply(lambda x: not isinstance(x, dict) or not x)
        ].index
        if unevaluated_indices.empty:
            logger.info(
                "No unevaluated responses found in the current DataFrame for batch processing."
            )
            return df

        unevaluated_df = df.loc[unevaluated_indices]
        logger.info(f"Found {len(unevaluated_df)} responses for batch evaluation.")

        evaluation_count = 0
        # Process in batches
        for start in range(0, len(unevaluated_df), batch_size):
            batch = unevaluated_df.iloc[start : start + batch_size]

            for idx, row in batch.iterrows():
                try:
                    email = row[timestamp_col]  # Assuming Timestamp holds the email/ID
                    logger.debug(f"Evaluating response for {email}")
                    # Run evaluation
                    evaluation = run_evaluation(
                        llm,
                        evaluation_prompt,
                        row[PART1_COL],  # Use named column
                        row[PART2_COL],  # Use named column
                    )

                    if evaluation:
                        # Save results using the database function
                        save_success = save_enriched_data(
                            email=email,
                            ai_output=evaluation,  # Pass the dict directly
                            teacher_notes=row.get(
                                "teacher_notes", ""
                            ),  # Preserve existing notes
                            teacher_score=row.get(
                                "teacher_score", None
                            ),  # Preserve teacher_score
                            finalized=row.get(
                                "finalized", False
                            ),  # Preserve finalized status
                        )
                        if save_success:
                            logger.info(f"Successfully saved evaluation for {email}")
                            # Update the DataFrame in memory immediately using the unique email/timestamp
                            df.loc[df[timestamp_col] == email, "ai_evaluation"] = [
                                evaluation
                            ]  # Use list for assignment with boolean index
                            df.loc[df[timestamp_col] == email, "last_modified"] = (
                                datetime.now()
                            )
                            evaluation_count += 1
                        else:
                            logger.error(f"Failed to save evaluation for {email}")

                except Exception as e:
                    # Log error for the specific row and continue
                    logger.error(
                        f"Error evaluating or saving response for {row.get(timestamp_col, 'UNKNOWN')}: {e}"
                    )
                    continue

        logger.info(
            f"Batch evaluation complete. Evaluated {evaluation_count} responses."
        )
        return df  # Return the updated DataFrame

    except Exception as e:
        logger.error(f"Batch evaluation failed: {e}")
        return df  # Return original DataFrame on outer error


st.title("✨🦔 AI Response Evaluator")

# --- Session State Initialization ---
if "initial_load_complete" not in st.session_state:
    st.session_state.initial_load_complete = False
if "selected_response" not in st.session_state:
    st.session_state.selected_response = None  # Initialize if not present
if "confirm_rerun" not in st.session_state:
    st.session_state.confirm_rerun = False  # Initialize confirmation flag


# --- Callback Functions ---
def set_selection_state(new_response_label):
    """Callback to update selection state only."""
    logger.debug(f"Callback set_selection_state called with: {new_response_label}")
    if new_response_label:
        st.session_state.selected_response = new_response_label
    else:
        # Clear selection state if None is passed
        if "selected_response" in st.session_state:
            del st.session_state["selected_response"]
    # Reset confirmation state on any navigation/selection change
    st.session_state.confirm_rerun = False


def sync_query_param():
    """Callback to sync query param FROM session state."""
    logger.debug("Callback sync_query_param called.")
    if "selected_response" in st.session_state:
        new_val = st.session_state.selected_response
        if st.query_params.get("response") != new_val:
            logger.debug(f"Syncing QP to: {new_val}")
            st.query_params["response"] = new_val
    else:
        # If selection state is cleared, clear query param
        query_params = st.query_params.to_dict()
        if "response" in query_params:
            logger.debug("Syncing QP: Clearing response param.")
            del query_params["response"]
            st.query_params.from_dict(query_params)


def save_and_finalize(email, ai_eval_data, notes, score):
    """Callback to save data and mark as finalized."""
    logger.debug(f"Callback save_and_finalize called for: {email}")
    save_success = save_enriched_data(
        email,
        ai_eval_data,
        notes,
        score,
        True,  # Force finalize
    )
    if save_success:
        st.success("🎉 Marked as complete!")
        st.rerun()  # Force a rerun immediately after successful save
        return True  # Indicate success
    else:
        st.error("Failed to save finalization.")
        return False  # Indicate failure


# Initialize query params and sync session state if needed (Revised)
current_response_qp = st.query_params.get("response", None)
# On initial load or refresh, if QP exists but state doesn't, sync state FROM QP
if "selected_response" not in st.session_state and current_response_qp:
    logger.debug(f"Initial load: Setting session state from QP: {current_response_qp}")
    st.session_state.selected_response = current_response_qp
# If state exists but QP doesn't (e.g. browser back button cleared QP), sync QP FROM state
elif "selected_response" in st.session_state and not current_response_qp:
    logger.debug("QP missing: Setting QP from session state.")
    st.query_params["response"] = st.session_state.selected_response
# We no longer force state = qp if they differ during a run,
# as on_change callback handles user interaction driving the state.

# Use the value from session state as the source of truth during the run
current_response = st.session_state.get("selected_response", None)

# --- Initialize LLM (only if prompt loaded) ---
llm = None
try:
    llm = AzureChatOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("OPENAI_API_VERSION", "2024-10-21"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
        azure_endpoint=os.getenv(
            "AZURE_OPENAI_ENDPOINT", "https://eddo-openai.openai.azure.com/"
        ),
    )
    logger.info("AzureChatOpenAI model initialized.")
except Exception as e:
    st.error(f"Failed to initialize AzureChatOpenAI model: {e}")
    logger.error(f"Failed to initialize AzureChatOpenAI model: {e}")

# Add debug expander
if logger.getEffectiveLevel() <= logging.DEBUG:
    with st.expander("Debug Information"):
        st.write("Current Working Directory:", os.getcwd())
        st.write(
            "Environment Variables:",
            {k: v for k, v in os.environ.items() if "sheet" in k.lower()},
        )
        st.write("Logger Level:", logging.getLevelName(logger.getEffectiveLevel()))
        st.write("Logger Handlers:", [type(h).__name__ for h in logger.handlers])

# Load responses and enriched data
df_raw = get_google_sheet()

if df_raw is not None:
    # --- Rename long column name (Option 2) ---
    NEW_PART2_NAME = "Part 2 Explanation"  # Define the new shorter name
    if PART2_COL in df_raw.columns:
        df_raw.rename(columns={PART2_COL: NEW_PART2_NAME}, inplace=True)
        logger.info(f'Renamed column "{PART2_COL[:30]}..." to "{NEW_PART2_NAME}"')
        PART2_COL = (
            NEW_PART2_NAME  # Re-add this line: Update constant for subsequent use
        )
    else:
        logger.warning(f'Column "{PART2_COL[:30]}..." not found for renaming.')
    # --- End Renaming ---

    if df_raw.empty:
        logger.warning("No responses found!")
    else:
        # Merge with enriched data
        df = get_merged_data(df_raw)
        logger.info(f"Loaded {len(df)} responses successfully!")

        # Show success message
        st.sidebar.success(f"📝 {len(df)} responses loaded from Google Forms")

        # Calculate progress metrics
        total_responses = len(df)
        # Count rows where ai_evaluation is a non-empty dict
        evaluated_responses = (
            df["ai_evaluation"].apply(lambda x: isinstance(x, dict) and bool(x)).sum()
        )
        finalized_responses = df["finalized"].fillna(False).sum()
        # Correctly calculate not started based on evaluated
        # unevaluated_responses = total_responses - evaluated_responses
        not_started_responses = (
            total_responses - evaluated_responses - finalized_responses
        )  # More precise

        # Filtering section
        st.sidebar.header("Filter Responses")

        # Try different possible column names for teacher
        teacher_col = next(
            (
                col
                for col in ["Teacher Name", "Teacher", "teacher_name", "teacher"]
                if col in df.columns
            ),
            None,
        )
        if teacher_col:
            teachers = ["All"] + sorted(df[teacher_col].unique().tolist())
            selected_teacher = st.sidebar.selectbox("Teacher:", teachers)
        else:
            selected_teacher = "All"

        # Try different possible column names for hour
        hour_col = next(
            (
                col
                for col in ["Hour", "Period", "Class Period", "hour"]
                if col in df.columns
            ),
            None,
        )
        if hour_col:
            hours = ["All"] + sorted(df[hour_col].unique().tolist())
            selected_hour = st.sidebar.selectbox("Hour:", hours)
        else:
            selected_hour = "All"

        # Add status filter
        status_options = ["All", "Not Started", "In Progress", "Finalized"]
        selected_status = st.sidebar.selectbox("Review Status:", status_options)

        # Apply filters
        filtered_df = df.copy()
        if selected_teacher != "All" and teacher_col:
            filtered_df = filtered_df[filtered_df[teacher_col] == selected_teacher]
        if selected_hour != "All" and hour_col:
            filtered_df = filtered_df[filtered_df[hour_col] == selected_hour]
        if selected_status != "All":
            if selected_status == "Not Started":
                filtered_df = filtered_df[filtered_df["ai_evaluation"].isna()]
            elif selected_status == "In Progress":
                filtered_df = filtered_df[
                    (filtered_df["ai_evaluation"].notna())
                    & (~filtered_df["finalized"].fillna(False))
                ]
            elif selected_status == "Finalized":
                filtered_df = filtered_df[filtered_df["finalized"].fillna(False)]

        # --- Add Debugging ---
        if logger.getEffectiveLevel() <= logging.DEBUG:
            logger.debug(f"Filtered DataFrame Length: {len(filtered_df)}")
            if not filtered_df.empty:
                logger.debug(
                    "Filtered DataFrame Head:\n" + filtered_df.head().to_string()
                )
        # --- End Debugging ---

        # Progress metrics for filtered data
        st.sidebar.divider()

        # Add a test write here
        # if logger.getEffectiveLevel() <= logging.DEBUG:
        #     st.write("DEBUG: About to process columns and set up sidebar...")

        # Calculate metrics for filtered data
        filtered_total = len(filtered_df)
        # Count rows where ai_evaluation is a non-empty dict
        filtered_evaluated = (
            filtered_df["ai_evaluation"]
            .apply(lambda x: isinstance(x, dict) and bool(x))
            .sum()
        )
        filtered_finalized = filtered_df["finalized"].fillna(False).sum()
        # Correctly calculate not started/in progress for filtered data
        filtered_in_progress = filtered_evaluated - filtered_finalized
        filtered_not_started = (
            filtered_total - filtered_in_progress - filtered_finalized
        )

        # Combined section for filter summary and response selection
        st.sidebar.markdown(
            f"""
        ### 📝 Responses
        {filtered_total} total • ⚪️ {filtered_not_started} • 🟡 {filtered_in_progress} • 🟢 {filtered_finalized}
        """
        )

        # Add batch evaluation button if there are unevaluated responses
        # Use the new filtered_not_started count
        # filtered_unevaluated = len(filtered_df[filtered_df["ai_evaluation"].isna()])

        # Main content area
        # Show response details only if a response is selected AND initial load is complete
        if selected_response_label and st.session_state.get(
            "initial_load_complete", False
        ):
            try:
                # Extract email from the selected label (split by backtick)
                # Example format: "🟢 Student Name (`student@example.com`)"
                parts = selected_response_label.split("`")
                if len(parts) >= 2:
                    selected_email = parts[-2]
                else:
                    raise ValueError("Could not parse email from selection label")

                selected_row = filtered_df[
                    filtered_df[email_col] == selected_email
                ].iloc[0]
                current_idx = response_options.index(selected_response_label)

                # Status indicator
                status_col1, status_col2 = st.columns([3, 1])
                with status_col1:
                    if (
                        pd.notna(selected_row.get("finalized"))
                        and selected_row["finalized"]
                    ):
                        st.success("✅ Review Complete")
                    elif pd.notna(selected_row.get("ai_evaluation")):
                        st.warning("🔄 Review In Progress")
                    else:
                        st.info("🆕 Not Yet Evaluated")
                with status_col2:
                    if pd.notna(selected_row.get("last_modified")):
                        st.caption(
                            f"Last updated: {pd.to_datetime(selected_row['last_modified']).strftime('%Y-%m-%d %H:%M')}"
                        )

                # Create expandable section for student info
                with st.expander("📝 Student Information", expanded=False):
                    info_col1, info_col2, info_col3 = st.columns([2, 1, 1])
                    with info_col1:
                        st.markdown(f"**Name:** {selected_row[name_col]}")
                        st.markdown(f"**Email:** {selected_row[email_col]}")
                    with info_col2:
                        st.markdown(f"**Teacher:** {selected_row['Teacher Name']}")
                    with info_col3:
                        st.markdown(f"**Hour:** {selected_row['Hour']}")

                # Response content in expander
                has_evaluation = pd.notna(selected_row.get("ai_evaluation"))
                prev_eval = None  # Initialize prev_eval here
                with st.expander("📝 Student Response", expanded=not has_evaluation):
                    # Only show Part 1 if it exists
                    part1_content = selected_row.get(PART1_COL, "")
                    if pd.notna(part1_content) and part1_content.strip():
                        st.markdown("**Part 1 (Optional):**")
                        st.text_area(
                            "Part 1",
                            part1_content,
                            height=150,
                            disabled=True,
                            label_visibility="collapsed",
                            key="part1_display",
                        )

                st.markdown("**Part 2:**")
                st.text_area(
                    "Part 2",
                    selected_row.get(PART2_COL, ""),
                    height=150,
                    disabled=True,
                    label_visibility="collapsed",
                    key="part2_display",
                )

                # Show existing evaluation if available
                if has_evaluation:
                    try:
                        # Directly use the dictionary from the DataFrame
                        evaluation_data = selected_row["ai_evaluation"]
                        if isinstance(evaluation_data, dict):
                            prev_eval = evaluation_data
                        elif (
                            isinstance(evaluation_data, str) and evaluation_data.strip()
                        ):
                            # Fallback: try parsing if it's somehow still a string
                            prev_eval = json.loads(evaluation_data)
                        else:
                            logger.warning(
                                f"AI evaluation data for {selected_email} is not a valid dict or JSON string: {evaluation_data}"
                            )
                            st.warning(
                                "Could not display previous AI evaluation data (invalid format)."
                            )

                        # Check if prev_eval was successfully assigned
                        if prev_eval:
                            st.markdown("### 🤖 AI Evaluation")

                            # Show annotated response first
                            with st.expander("📝 Annotated Response", expanded=True):
                                st.markdown(prev_eval.get("annotated_response", ""))

                            # Show feedback
                            with st.expander("💭 Feedback", expanded=True):
                                feedback_data = prev_eval.get("feedback", {})
                                st.markdown("**Strengths:**")
                                st.markdown(feedback_data.get("strengths", ""))
                                st.markdown("**Suggestions:**")
                                st.markdown(feedback_data.get("suggestions", ""))
                        else:
                            # Handle case where prev_eval couldn't be loaded/parsed
                            if (
                                not isinstance(evaluation_data, dict)
                                and evaluation_data
                            ):
                                st.error(
                                    "Could not load previous evaluation data (invalid format)."
                                )

                    except Exception as e:
                        st.error("Could not load previous evaluation data")
                        logger.error(f"Error loading evaluation data: {e}")

                # Display AI scores just before teacher evaluation
                # Check if prev_eval is a valid dictionary before proceeding
                if prev_eval and isinstance(prev_eval, dict):
                    try:
                        scores = prev_eval.get("rubric_scores", {})

                        # Custom CSS for score section
                        st.markdown(
                            """
                            <style>
                                .score-label { 
                                    color: #555; 
                                    font-size: 0.9rem; 
                                    font-weight: 600; 
                                    margin-bottom: 0.2rem; 
                                }
                                .score-value { 
                                    font-size: 1.8rem; 
                                    font-weight: 700; 
                                    color: #0F52BA;
                                    margin: 0;
                                    line-height: 1;
                                }
                                .score-denominator {
                                    color: #666;
                                    font-size: 1rem;
                                    font-weight: 400;
                                }
                                .overall-score {
                                    font-size: 2.2rem;
                                    color: #1e3d8f;
                                }
                            </style>
                        """,
                            unsafe_allow_html=True,
                        )

                        st.markdown("#### ✨🦔 AI Rubric Scores")

                        # Create container for scores
                        score_container = st.container()
                        with score_container:
                            # Create two columns for scores with minimal spacing
                            criteria_col, overall_col = st.columns([2, 1])

                            # Individual criteria scores in left column
                            with criteria_col:
                                # Create a 2x2 grid for criteria scores
                                row1_cols = st.columns(2)
                                with row1_cols[0]:
                                    st.markdown(
                                        '<p class="score-label">SEP</p>',
                                        unsafe_allow_html=True,
                                    )
                                    st.markdown(
                                        f'<p class="score-value">{scores.get("SEP", 0)}<span class="score-denominator">/4</span></p>',
                                        unsafe_allow_html=True,
                                    )
                                with row1_cols[1]:
                                    st.markdown(
                                        '<p class="score-label">DCI</p>',
                                        unsafe_allow_html=True,
                                    )
                                    st.markdown(
                                        f'<p class="score-value">{scores.get("DCI", 0)}<span class="score-denominator">/4</span></p>',
                                        unsafe_allow_html=True,
                                    )

                                row2_cols = st.columns(2)
                                with row2_cols[0]:
                                    st.markdown(
                                        '<p class="score-label">CCC</p>',
                                        unsafe_allow_html=True,
                                    )
                                    st.markdown(
                                        f'<p class="score-value">{scores.get("CCC", 0)}<span class="score-denominator">/4</span></p>',
                                        unsafe_allow_html=True,
                                    )
                                with row2_cols[1]:
                                    st.markdown(
                                        '<p class="score-label">Communication</p>',
                                        unsafe_allow_html=True,
                                    )
                                    st.markdown(
                                        f'<p class="score-value">{scores.get("Communication", 0)}<span class="score-denominator">/4</span></p>',
                                        unsafe_allow_html=True,
                                    )

                            # Overall score in right column with divider
                            with overall_col:
                                st.markdown(
                                    '<p class="score-label">✨Overall</p>',
                                    unsafe_allow_html=True,
                                )
                                st.markdown(
                                    f'<p class="score-value overall-score">{scores.get("Overall", 0)}<span class="score-denominator">/4</span></p>',
                                    unsafe_allow_html=True,
                                )

                    except Exception as e:
                        st.error("Could not display AI scores")
                        logger.error(f"Error displaying AI scores: {e}")

                # Teacher evaluation section
                st.markdown("### 👩‍🏫 Teacher Evaluation")

                # Create two columns for score and notes
                score_col, notes_col = st.columns([1, 3])

                with score_col:
                    st.markdown("**Score**")
                    current_score = selected_row.get("teacher_score")
                    # Convert to int if exists and valid, otherwise default to None
                    default_score = (
                        int(float(current_score)) if pd.notna(current_score) else None
                    )
                    # Create list of scores in descending order
                    score_options = [4, 3, 2, 1]
                    teacher_score = st.radio(
                        "Score",
                        options=score_options,
                        index=(
                            None
                            if default_score is None
                            else score_options.index(default_score)
                        ),
                        help="Select score from 4-0",
                        label_visibility="collapsed",
                    )
                    if teacher_score is None:
                        st.caption("⚠️ Select a score")
                    else:
                        st.caption(f"Selected: {teacher_score}/4")

                with notes_col:
                    current_notes = selected_row.get("teacher_notes", "")
                    if pd.isna(current_notes):
                        current_notes = ""
                    teacher_notes = st.text_area(
                        "Evaluation Notes",
                        value=current_notes,
                        height=100,
                        placeholder="Add your evaluation notes, observations, or feedback...",
                        key="teacher_notes",
                    )
                    st.caption(f"{len(teacher_notes)}/500 characters")

                # Navigation controls at bottom
                st.divider()
                nav_cols = st.columns(4)

                # Back button
                if current_idx > 0:
                    back_target = response_options[current_idx - 1]
                    nav_cols[0].button(
                        "← Back",
                        use_container_width=True,
                        on_click=set_selection_state,  # Use callback
                        args=(back_target,),  # Pass target label
                    )
                    # Remove direct state/QP update and rerun from here

                # Evaluate/Rerun button
                rerun_button = False  # Initialize
                if llm and evaluation_prompt:
                    has_eval = pd.notna(selected_row.get("ai_evaluation"))
                    button_label = "🔄 Rerun" if has_eval else "🤖 Evaluate"
                    rerun_button = nav_cols[1].button(
                        button_label,
                        type="secondary",
                        use_container_width=True,
                        key="eval_rerun_button",
                    )
                    # Add confirmation only for rerun
                    if rerun_button and has_eval:
                        if not st.session_state.get("confirm_rerun", False):
                            st.session_state.confirm_rerun = True
                            rerun_button = False  # Don't proceed yet
                            st.rerun()  # Rerun to show checkbox

                        if st.session_state.get("confirm_rerun", False):
                            if nav_cols[1].checkbox(
                                "✓ Confirm rerun", key="rerun_confirm_box"
                            ):
                                st.session_state.confirm_rerun = (
                                    False  # Reset on confirm
                                )
                                # Let rerun_button remain True
                            else:
                                rerun_button = False  # Checkbox not ticked
                        elif rerun_button and not has_eval:
                            st.session_state.confirm_rerun = (
                                False  # Reset if Evaluate is clicked
                            )

                # Skip button
                if current_idx < len(response_options) - 1:
                    skip_target = response_options[current_idx + 1]
                    nav_cols[2].button(
                        "Skip →",
                        use_container_width=True,
                        on_click=set_selection_state,  # Use callback
                        args=(skip_target,),  # Pass target label
                    )
                    # Remove direct state/QP update and rerun from here

                # Done button
                # Enable Done button only if teacher_score is selected
                done_disabled = teacher_score is None
                done_help = (
                    "Select a teacher score before marking as Done."
                    if done_disabled
                    else None
                )
                # Ensure ai_eval_data is defined before being passed to callback args
                ai_eval_data = None
                current_ai_eval = selected_row.get("ai_evaluation")
                if pd.notna(current_ai_eval):
                    if isinstance(current_ai_eval, dict):
                        ai_eval_data = current_ai_eval  # It's already a dict
                    elif isinstance(current_ai_eval, str) and current_ai_eval.strip():
                        try:
                            ai_eval_data = json.loads(current_ai_eval)
                        except (json.JSONDecodeError, TypeError):
                            logger.warning(
                                f"Could not parse existing ai_evaluation string for {selected_email} when setting up Done button args: {current_ai_eval}"
                            )
                            ai_eval_data = (
                                current_ai_eval  # Pass raw string if parsing fails
                            )
                    else:
                        # Handle other non-dict, non-string types if necessary, or just pass None
                        logger.warning(
                            f"Unexpected type for ai_evaluation for {selected_email}: {type(current_ai_eval)}"
                        )
                        ai_eval_data = None

                # Done Button - Now define it *after* ai_eval_data is set
                if nav_cols[3].button(
                    "Done ✨",
                    type="primary",
                    use_container_width=True,
                    disabled=done_disabled,
                    help=done_help,
                    on_click=save_and_finalize,  # Use callback
                    args=(  # Pass necessary args to callback
                        selected_email,
                        ai_eval_data,
                        teacher_notes,
                        teacher_score,
                    ),
                ):
                    # This block now executes *after* the callback runs (due to rerun)
                    # We need to check if finalization was successful to navigate.
                    # The navigation logic needs to be separate or triggered by the callback's effect.
                    pass

                # Run evaluation if requested
                if llm and evaluation_prompt and rerun_button:
                    with st.spinner("Running evaluation..."):
                        try:
                            st.session_state.confirm_rerun = (
                                False  # Reset confirm after trigger
                            )
                            logger.info(
                                f"Starting evaluation/rerun process for {selected_email}"
                            )
                            evaluation = run_evaluation(
                                llm,
                                evaluation_prompt,
                                selected_row.get(PART1_COL, ""),
                                selected_row.get(PART2_COL, ""),
                            )
                            logger.debug(
                                f"Evaluation result for {selected_email}: {evaluation}"
                            )

                            # Save the evaluation result
                            if evaluation:
                                # Use the current teacher score and notes if available
                                save_success = save_enriched_data(
                                    selected_email,
                                    evaluation,  # Pass the new evaluation dict
                                    teacher_notes,
                                    teacher_score,  # Pass current teacher score
                                    selected_row.get(
                                        "finalized", False
                                    ),  # Keep existing finalized status
                                )
                                if save_success:
                                    st.success("Evaluation saved successfully!")
                                    st.rerun()
                                else:
                                    st.error("Failed to save evaluation")
                            else:
                                st.error("Evaluation returned no results")

                        except Exception as e:
                            st.error(f"Error running evaluation: {str(e)}")
                            logger.error(f"Evaluation error: {str(e)}")
                            logger.error(
                                f"Selected row columns: {selected_row.index.tolist()}"
                            )

            except (IndexError, ValueError) as e:
                st.error(
                    "Could not find the selected response. Ensure emails are unique."
                )
                logger.error(f"Error loading response: {e}")
        else:
            # Welcome/instruction state (shows on initial load or if no response selected)
            st.info(
                "👈 Find your students by selecting your name and class period in the sidebar"
            )

            # Add helpful instructions
            st.markdown(
                """
            ### 🦎 Green Anole Assessment Review Tool

            All responses are loaded. Three steps to review:

            #### 1. Find Students
            Select your name and class period in the sidebar

            #### 2. Generate Feedback
            Click "Evaluate" to analyze responses (⚪️ → 🟡 → 🟢)

            #### 3. Review
            - Check AI feedback and scores
            - Add your score (1-4) and notes (Optional)
            - Click "Done" when finished
            - Results are tracked and you can export the data as a CSV

            """
            )

            # Show progress metrics if available
            if "df" in locals() and not df.empty:
                st.divider()
                st.markdown("### 📊 Current Progress")

                # Calculate metrics
                total = len(df)
                evaluated = (
                    df["ai_evaluation"]
                    .apply(lambda x: isinstance(x, dict) and bool(x))
                    .sum()
                )
                finalized = df["finalized"].fillna(False).sum()

                # Display metrics in columns
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Responses", total)
                with col2:
                    st.metric("Evaluated", f"{evaluated} ({evaluated/total*100:.1f}%)")
                with col3:
                    st.metric("Finalized", f"{finalized} ({finalized/total*100:.1f}%)")

            # Set the flag indicating the initial load process is complete
            st.session_state.initial_load_complete = True

else:
    logger.error("Failed to load responses")
    st.error("Failed to load responses")
