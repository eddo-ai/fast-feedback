import streamlit as st
import gspread
import pandas as pd
import re
from difflib import get_close_matches
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

# Load environment variables first
load_dotenv()

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger("streamlit")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(getattr(logging, log_level))
    logger.addHandler(handler)
logger.setLevel(getattr(logging, log_level))
logger.debug(f"Log level set to: {log_level}")

# --- Column detection configuration with regex and fuzzy matching ---
COLUMN_PATTERNS = {
    "timestamp": [r"^Timestamp$", r"^timestamp$"],
    "email": [r"Email Address", r"Email", r"email", r"Student Email"],
    "name": [r"Student Name", r"Name", r"Full Name", r"name", r"Student"],
    "teacher": [r"Teacher Name", r"Teacher", r"teacher_name", r"teacher"],
    "hour": [r"Hour", r"Period", r"Class Period", r"hour"],
    # Match any header starting with '1.' or containing key phrases
    "part1": [r"^1\.", r"notice about the behavior", r"body structures"],
    # Match headers starting with '2.' or containing 'Construct an explanation' or 'Part 2'
    "part2": [r"^2\.", r"Construct an explanation", r"Part 2"],
}


def generate_response_options(
    df: pd.DataFrame, email_col: str, name_col: str
) -> list[str]:
    """Generate a list of response options with status indicators.

    Args:
        df (pd.DataFrame): DataFrame containing response data
        email_col (str): Name of the email column
        name_col (str): Name of the student name column

    Returns:
        list[str]: List of formatted response options with status indicators
    """
    if df.empty or email_col not in df.columns or name_col not in df.columns:
        logger.warning(
            f"Could not generate response options. Columns '{email_col}' or '{name_col}' might be missing or DataFrame is empty."
        )
        if not df.empty:
            logger.debug(f"Available columns: {df.columns.tolist()}")
        return []

    options = []
    for _, row in df.sort_values(by=[name_col, email_col]).iterrows():
        email = row[email_col]
        name = row.get(name_col, "Unknown Name")

        # Determine status indicator
        if row.get("finalized", False):
            status = "🟢"  # Finalized
        elif isinstance(row.get("ai_evaluation"), dict) and row.get("ai_evaluation"):
            status = "🟡"  # In progress
        else:
            status = "⚪️"  # Not started

        options.append(f"{status} {name} (`{email}`)")

    return options


def detect_and_rename(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects and renames sheet columns to our short internal names.
    Uses exact, regex, fuzzy matching, and interactive fallback.
    """
    cols = df.columns.tolist()
    mapping = {}
    missing = []

    # 1) Try exact, regex, then fuzzy match
    for internal, patterns in COLUMN_PATTERNS.items():
        match = None
        # exact
        for pat in patterns:
            if pat in cols:
                match = pat
                break
        # regex
        if not match:
            for pat in patterns:
                rx = re.compile(pat, re.IGNORECASE)
                for c in cols:
                    if rx.search(c):
                        match = c
                        break
                if match:
                    break
        # fuzzy
        if not match:
            cand = get_close_matches(internal, cols, n=1, cutoff=0.6)
            if cand:
                match = cand[0]
        if match:
            mapping[match] = internal
            logger.info(f"Mapped column '{match}' to '{internal}'")
        else:
            missing.append(internal)

    # 2) Interactive fallback
    if missing:
        st.sidebar.warning(
            f"Could not auto-detect columns: {missing}\nPlease select manually."
        )
        for internal in missing:
            choice = st.sidebar.selectbox(
                f"Which column is '{internal}'?",
                [c for c in cols if c not in mapping.keys()],
                key=f"col_{internal}",
            )
            if choice:
                mapping[choice] = internal
                logger.info(f"Manually mapped column '{choice}' to '{internal}'")

    # Apply the mapping
    renamed_df = df.rename(columns=mapping)

    # Update global column references based on mapping
    global PART1_COL, PART2_COL
    PART1_COL = "part1"  # Use the internal name
    PART2_COL = "part2"  # Use the internal name

    return renamed_df


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
    """Connect to Google Sheets and return the worksheet data with normalized column names"""
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

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Debug original columns
        logger.info("=== Original DataFrame Columns ===")
        for col in df.columns:
            logger.info(f"  - {col}")

        # Apply our new column detection and renaming
        df = detect_and_rename(df)

        # Debug renamed columns
        logger.info("=== Renamed DataFrame Columns ===")
        for col in df.columns:
            logger.info(f"  - {col}")

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


def get_merged_data(raw_df, email_col):
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

        # Check if the detected email column exists in raw_df before merging
        if email_col not in raw_df.columns:
            logger.error(
                f"Detected email column '{email_col}' not found in raw data. Cannot merge."
            )
            # Optionally, add default enriched columns to raw_df and return
            for col in [
                "ai_evaluation",
                "teacher_notes",
                "finalized",
                "last_modified",
                "teacher_score",
            ]:
                if col not in raw_df.columns:
                    raw_df[col] = pd.NA
            raw_df["ai_evaluation"] = raw_df["ai_evaluation"].fillna({})
            raw_df["teacher_notes"] = raw_df["teacher_notes"].fillna("")
            raw_df["finalized"] = raw_df["finalized"].fillna(False).astype(bool)
            raw_df["last_modified"] = pd.to_datetime(
                raw_df["last_modified"].fillna(pd.NaT)
            )
            if "teacher_score" in raw_df.columns:
                raw_df["teacher_score"] = pd.to_numeric(
                    raw_df["teacher_score"], errors="coerce"
                )
            return raw_df

        # Merge the dataframes - Use the detected email_col
        merged_df = pd.merge(
            raw_df,
            enriched_df,
            # Use detected email from raw data and the key (which is email) from enriched data
            left_on=email_col,  # Use detected email column
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
        # Add default enriched columns if merge fails completely
        for col in [
            "ai_evaluation",
            "teacher_notes",
            "finalized",
            "last_modified",
            "teacher_score",
        ]:
            if col not in raw_df.columns:
                raw_df[col] = pd.NA
        raw_df["ai_evaluation"] = raw_df["ai_evaluation"].fillna({})
        raw_df["teacher_notes"] = raw_df["teacher_notes"].fillna("")
        raw_df["finalized"] = raw_df["finalized"].fillna(False).astype(bool)
        raw_df["last_modified"] = pd.to_datetime(raw_df["last_modified"].fillna(pd.NaT))
        if "teacher_score" in raw_df.columns:
            raw_df["teacher_score"] = pd.to_numeric(
                raw_df["teacher_score"], errors="coerce"
            )
        return raw_df  # Return original data with defaults on error


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


def batch_evaluate(df, llm, evaluation_prompt, email_col, batch_size=5):
    """Simple batch evaluation using LLM's built-in batching."""
    try:
        # Get unevaluated responses based on ai_evaluation being null/empty in the DataFrame
        # Use the dynamically detected email_col as the unique key
        if email_col not in df.columns:
            logger.error(
                f"Email column '{email_col}' not found in DataFrame for batch evaluation."
            )
            return df

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
                    email = row[email_col]  # Use detected email column
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
                            # Update the DataFrame in memory immediately using the unique email
                            df.loc[df[email_col] == email, "ai_evaluation"] = [
                                evaluation
                            ]  # Use detected email_col  # Use list for assignment with boolean index
                            df.loc[df[email_col] == email, "last_modified"] = (
                                datetime.now()
                            )  # Use detected email_col
                            evaluation_count += 1
                        else:
                            logger.error(f"Failed to save evaluation for {email}")

                except Exception as e:
                    # Log error for the specific row and continue
                    logger.error(
                        f"Error evaluating or saving response for {row.get(email_col, 'UNKNOWN')}: {e}"  # Use detected email_col
                    )
                    continue

        logger.info(
            f"Batch evaluation complete. Evaluated {evaluation_count} responses."
        )
        return df  # Return the updated DataFrame

    except Exception as e:
        logger.error(f"Batch evaluation failed: {e}")
        return df  # Return original DataFrame on outer error


# --- Helper Function for Column Detection ---
def find_column(df_columns, potential_names):
    """Find the first matching column name from a list of potential names."""
    for name in potential_names:
        if name in df_columns:
            logger.info(f"Detected column: '{name}'")
            return name
    logger.warning(f"Could not find any of the potential columns: {potential_names}")
    return None


# --- Streamlit App ---
st.title("✨🦔 AI Response Evaluator")

# --- Session State Initialization ---
if "initial_load_complete" not in st.session_state:
    st.session_state.initial_load_complete = False
if "selected_response" not in st.session_state:
    st.session_state.selected_response = None  # Initialize if not present
if "confirm_rerun" not in st.session_state:
    st.session_state.confirm_rerun = False  # Initialize confirmation flag


# --- Callback Functions ---
def set_selection_state(target=None):
    """Callback to update selection state. Gets value directly from session state."""
    logger.debug(
        f"Callback set_selection_state called with value from selectbox: {st.session_state.response_selector}"
    )
    if target is not None:
        st.session_state.response_selector = target
    st.session_state.selected_response = st.session_state.response_selector
    # Reset confirmation state on any navigation/selection change
    st.session_state.confirm_rerun = False


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


def clear_teacher_evaluation():
    """Callback to clear teacher score and notes."""
    # Get current email from selected response
    if st.session_state.selected_response:
        try:
            email = st.session_state.selected_response.split("`")[
                1
            ]  # Extract email from format
            # Clear the specific fields in session state
            st.session_state[f"teacher_score_{email}"] = None
            st.session_state[f"teacher_notes_{email}"] = ""
            # Save the cleared state to database
            save_enriched_data(
                email=email,
                ai_output=st.session_state.get(
                    "prev_eval"
                ),  # Keep existing AI evaluation
                teacher_notes="",
                teacher_score=None,
                finalized=False,  # Unfinalize when clearing
            )
            st.rerun()
        except Exception as e:
            logger.error(f"Error clearing teacher score: {e}")


# Use the value from session state as the source of truth during the run
# current_response = st.session_state.get("selected_response", None) # Determined later after sync

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
    # Optionally stop if LLM is crucial
    # st.stop()


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
    # --- Dynamic Column Detection ---
    email_col = "email"  # Use internal names
    name_col = "name"
    teacher_col = "teacher"
    hour_col = "hour"

    # Check if essential columns were found
    if not email_col or not name_col:
        st.error(
            "Could not automatically detect required 'Email' or 'Student Name' columns. Please check the Google Sheet column headers and the `COLUMN_PATTERNS` dictionary in the script."
        )
        st.stop()  # Stop execution if essential columns are missing

    if df_raw.empty:
        logger.warning("No responses found!")
    else:
        # Merge with enriched data - pass detected email_col
        df = get_merged_data(df_raw, email_col)
        logger.info(f"Loaded {len(df)} responses successfully!")

        # --- State Synchronization (Session State & Query Params) ---
        # 1. Get potential values from session state and query params
        state_response = st.session_state.get("selected_response", None)
        qp_response = st.query_params.get("response", None)

        # 2. Generate response options first so we can validate against them
        response_options = generate_response_options(df, email_col, name_col)

        # 3. Validate current selection against available options
        current_response = None
        if state_response and state_response in response_options:
            current_response = state_response
        elif qp_response and qp_response in response_options:
            current_response = qp_response
            st.session_state.selected_response = (
                current_response  # Update session state
            )

        # 4. Clear invalid selections
        if state_response and state_response not in response_options:
            logger.warning(
                f"Selected response '{state_response}' not found in current filtered options. Clearing selection."
            )
            if "selected_response" in st.session_state:
                del st.session_state["selected_response"]

        if qp_response and qp_response != current_response:
            logger.debug(
                "Clearing query param because it doesn't match current selection"
            )
            st.query_params.clear()

        # Now `current_response` holds the definitive selected response for this run
        logger.debug(f"Final current_response for this run: {current_response}")

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

        # Use detected column names here
        selected_teacher = "All"
        if teacher_col:
            teachers = ["All"] + sorted(df[teacher_col].unique().tolist())
            selected_teacher = st.sidebar.selectbox("Teacher:", teachers)
        else:
            logger.info("Teacher column not detected, filter disabled.")

        # Use detected column names here
        selected_hour = "All"
        if hour_col:
            hours = ["All"] + sorted(df[hour_col].unique().tolist())
            selected_hour = st.sidebar.selectbox("Hour:", hours)
        else:
            logger.info("Hour column not detected, filter disabled.")

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
                # More robust check for "Not Started" (handles None, NaN, empty dicts)
                filtered_df = filtered_df[
                    filtered_df["ai_evaluation"].apply(
                        lambda x: not isinstance(x, dict) or not x
                    )
                    & (
                        ~filtered_df["finalized"].fillna(False)
                    )  # Ensure not finalized either
                ]
            elif selected_status == "In Progress":
                # Check for non-empty dict in ai_evaluation AND not finalized
                filtered_df = filtered_df[
                    filtered_df["ai_evaluation"].apply(
                        lambda x: isinstance(x, dict) and bool(x)
                    )
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

        # Find the index of the current selection in the options list
        current_selection_index = None
        if current_response and response_options:
            try:
                current_selection_index = response_options.index(current_response)
            except ValueError:
                logger.warning(
                    f"Current response '{current_response}' not found in options list. This should not happen due to earlier validation."
                )
                current_selection_index = None

        # Add the radio list to choose a response
        st.sidebar.radio(
            "Select Response:",
            options=response_options,
            index=current_selection_index if current_selection_index is not None else 0,
            on_change=set_selection_state,  # Use the callback
            key="response_selector",  # Use a STATIC key
            label_visibility="collapsed",  # Hide label if desired
        )

        # Add Start Review button
        if (
            response_options and not current_response
        ):  # Only show when no response is selected
            first_response = response_options[0] if response_options else None
            if first_response:
                st.sidebar.button(
                    "✨ Start Review",
                    type="primary",
                    on_click=set_selection_state,
                    kwargs={"target": first_response},
                    use_container_width=True,
                )

        # Add batch evaluation callback
        def run_batch_evaluation():
            """Callback to run batch evaluation and ensure UI updates."""
            if not llm or not evaluation_prompt:
                st.error("AI evaluation is not available.")
                return

            # Get filtered unevaluated count
            filtered_not_started = int(st.session_state.get("filtered_not_started", 0))

            try:
                # Get the current filtered DataFrame from session state
                df = st.session_state.get("df", None)
                if df is None:
                    st.error("No data available for evaluation.")
                    return

                # Run batch evaluation
                df_updated = batch_evaluate(df, llm, evaluation_prompt, email_col)

                # Update session state with new DataFrame
                st.session_state.df = df_updated
                # Set a flag to indicate evaluation is complete
                st.session_state.batch_eval_complete = True

            except Exception as e:
                st.error(f"Error during batch evaluation: {str(e)}")
                logger.error(f"Batch evaluation error: {str(e)}")

        # Add batch evaluation button if there are unevaluated responses
        # Use the new filtered_not_started count
        if filtered_not_started > 0 and llm and evaluation_prompt:
            # Store the count in session state for the callback
            st.session_state.filtered_not_started = filtered_not_started

            st.sidebar.button(
                f"🤖 Evaluate {filtered_not_started} Responses",
                type="secondary",
                on_click=run_batch_evaluation,
                use_container_width=True,
            )

            # Check if batch evaluation just completed
            if st.session_state.get("batch_eval_complete", False):
                st.success("Batch evaluation completed!")
                # Clear the flag
                st.session_state.batch_eval_complete = False
                # Rerun here in the main flow
                st.rerun()

        # Main content area
        # Show response details only if a response is selected AND initial load is complete
        if current_response and st.session_state.get("initial_load_complete", False):
            try:
                # Extract email from the selected label (split by backtick)
                # Example format: "🟢 Student Name (`student@example.com`)"
                parts = current_response.split("`")
                if len(parts) >= 2:
                    selected_email = parts[-2]
                else:
                    raise ValueError("Could not parse email from selection label")

                # Find the selected row in the *filtered* DataFrame
                selected_rows = filtered_df[filtered_df[email_col] == selected_email]

                if selected_rows.empty:
                    # Handle case where selected response is no longer in filtered list
                    logger.warning(
                        f"Selected response {selected_email} not found in *filtered* data. Clearing selection."
                    )
                    st.warning(
                        "The previously selected response is no longer visible with the current filters. Please select another response."
                    )
                    if "selected_response" in st.session_state:
                        del st.session_state["selected_response"]
                    current_response = None
                    st.query_params.clear()
                    st.rerun()

                selected_row = selected_rows.iloc[0]
                # Find index in the potentially filtered response_options list
                current_idx = response_options.index(current_response)

                # Status indicator
                status_col1, status_col2 = st.columns([3, 1])
                with status_col1:
                    if (
                        pd.notna(selected_row.get("finalized"))
                        and selected_row["finalized"]
                    ):
                        st.success("✅ Review Complete")
                    elif (
                        pd.notna(selected_row.get("ai_evaluation"))
                        and isinstance(selected_row.get("ai_evaluation"), dict)
                        and selected_row.get("ai_evaluation")
                    ):
                        st.warning("🟡 Review In Progress")
                    else:
                        st.info("⚪️ Not Yet Evaluated")
                with status_col2:
                    if pd.notna(selected_row.get("last_modified")):
                        st.caption(
                            f"Last updated: {pd.to_datetime(selected_row['last_modified']).strftime('%Y-%m-%d %H:%M')}"
                        )

                # Create expandable section for student info
                with st.expander("📝 Student Information", expanded=False):
                    info_col1, info_col2, info_col3 = st.columns([2, 1, 1])
                    with info_col1:
                        # Use detected column names
                        st.markdown(f"**Name:** {selected_row.get(name_col, 'N/A')}")
                        st.markdown(f"**Email:** {selected_row.get(email_col, 'N/A')}")
                    with info_col2:
                        # Use detected teacher column if available
                        if teacher_col:
                            st.markdown(
                                f"**Teacher:** {selected_row.get(teacher_col, 'N/A')}"
                            )
                        else:
                            st.markdown("**Teacher:** (Not Detected)")
                    with info_col3:
                        # Use detected hour column if available
                        if hour_col:
                            st.markdown(
                                f"**Hour:** {selected_row.get(hour_col, 'N/A')}"
                            )
                        else:
                            st.markdown("**Hour:** (Not Detected)")

                # Response content in expander
                has_evaluation = (
                    pd.notna(selected_row.get("ai_evaluation"))
                    and isinstance(selected_row.get("ai_evaluation"), dict)
                    and selected_row.get("ai_evaluation")
                )
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
                            logger.warning(
                                f"AI evaluation for {selected_email} was a string, attempting parse."
                            )
                            prev_eval = json.loads(evaluation_data)
                        else:
                            # This case should be less likely now with the improved has_evaluation check
                            logger.warning(
                                f"AI evaluation data for {selected_email} is not a valid dict: {evaluation_data}"
                            )
                            st.warning(
                                "Could not display previous AI evaluation data (invalid format)."
                            )

                        # Check if prev_eval was successfully assigned
                        if prev_eval and isinstance(
                            prev_eval, dict
                        ):  # Ensure prev_eval is a dict
                            st.markdown("### 🤖 AI Evaluation")

                            # Show annotated response first
                            # Provide default empty string if key is missing
                            with st.expander("📝 Annotated Response", expanded=True):
                                st.markdown(
                                    prev_eval.get(
                                        "annotated_response",
                                        "*Annotation not found in evaluation data.*",
                                    )
                                )

                            # Show feedback
                            # Provide default empty dict/string if keys are missing
                            with st.expander("💭 Feedback", expanded=True):
                                feedback_data = prev_eval.get("feedback", {})
                                st.markdown("**Strengths:**")
                                st.markdown(
                                    feedback_data.get(
                                        "strengths", "*Strengths not found.*"
                                    )
                                )
                                st.markdown("**Suggestions:**")
                                st.markdown(
                                    feedback_data.get(
                                        "suggestions", "*Suggestions not found.*"
                                    )
                                )

                        else:
                            # This condition means has_evaluation was true, but prev_eval didn't become a dict
                            logger.error(
                                f"Evaluation data existed for {selected_email} but failed to load as dict: {evaluation_data}"
                            )
                            st.error(
                                "Could not load previous evaluation data despite it being present."
                            )

                    except json.JSONDecodeError as e:
                        st.error(
                            f"Could not parse previous evaluation data (JSON error): {e}"
                        )
                        logger.error(
                            f"Error parsing evaluation JSON for {selected_email}: {e}"
                        )
                    except Exception as e:
                        st.error(f"Could not load previous evaluation data: {e}")
                        logger.error(
                            f"Error loading evaluation data for {selected_email}: {e}"
                        )

                # Display AI scores just before teacher evaluation
                # Check if prev_eval is a valid dictionary before proceeding
                if prev_eval and isinstance(prev_eval, dict):
                    try:
                        # Provide default empty dict if key is missing
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
                                    # Provide default 0 if key missing
                                    st.markdown(
                                        f'<p class="score-value">{scores.get("SEP", 0)}<span class="score-denominator">/4</span></p>',
                                        unsafe_allow_html=True,
                                    )

                                with row1_cols[1]:
                                    st.markdown(
                                        '<p class="score-label">DCI</p>',
                                        unsafe_allow_html=True,
                                    )
                                    # Provide default 0 if key missing
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
                                    # Provide default 0 if key missing
                                    st.markdown(
                                        f'<p class="score-value">{scores.get("CCC", 0)}<span class="score-denominator">/4</span></p>',
                                        unsafe_allow_html=True,
                                    )

                                with row2_cols[1]:
                                    st.markdown(
                                        '<p class="score-label">Communication</p>',
                                        unsafe_allow_html=True,
                                    )
                                    # Provide default 0 if key missing
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
                                # Provide default 0 if key missing
                                st.markdown(
                                    f'<p class="score-value overall-score">{scores.get("Overall", 0)}<span class="score-denominator">/4</span></p>',
                                    unsafe_allow_html=True,
                                )

                            # Add rerun button below scores
                            rerun_button = False  # Initialize
                            if llm and evaluation_prompt:
                                button_label = (
                                    "🔄:hedgehog: Rerun Evaluation"
                                    if has_evaluation
                                    else "🤖 Evaluate with AI"
                                )
                                rerun_button_clicked = (
                                    st.button(  # Use different variable name
                                        button_label,
                                        type="secondary",
                                        use_container_width=True,
                                        key="eval_rerun_button",  # Keep key simple
                                    )
                                )
                                # Add confirmation only for rerun
                                if rerun_button_clicked and has_evaluation:
                                    if not st.session_state.get("confirm_rerun", False):
                                        st.session_state.confirm_rerun = True
                                        st.rerun()  # Rerun to show checkbox

                                    if st.session_state.get("confirm_rerun", False):
                                        confirm_key = f"rerun_confirm_{selected_email}"  # Unique key
                                        if st.checkbox(
                                            "✓ Confirm AI rerun", key=confirm_key
                                        ):
                                            st.session_state.confirm_rerun = (
                                                False  # Reset on confirm
                                            )
                                            rerun_button = True  # Set flag to proceed
                                        else:
                                            rerun_button = False  # Checkbox not ticked
                                            # Keep confirm_rerun = True until checkbox is ticked or user navigates away
                                    # No confirmation needed if it's the first evaluation
                                elif rerun_button_clicked and not has_evaluation:
                                    st.session_state.confirm_rerun = (
                                        False  # Reset just in case
                                    )
                                    rerun_button = True  # Set flag to proceed

                    except Exception as e:
                        st.error("Could not display AI scores")
                        logger.error(
                            f"Error displaying AI scores for {selected_email}: {e}"
                        )

                # Teacher evaluation section
                st.markdown("### 👩‍🏫 Teacher Score")

                # Add clear button above the evaluation fields
                if st.button(
                    "🗑️ Clear Score and Notes",
                    type="secondary",
                    help="Clear teacher score and notes",
                ):
                    clear_teacher_evaluation()

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
                    score_options_list = [4, 3, 2, 1]  # Renamed variable
                    score_index = None
                    if default_score is not None:
                        try:
                            score_index = score_options_list.index(default_score)
                        except ValueError:
                            logger.warning(
                                f"Saved teacher score {default_score} not in options {score_options_list}. Resetting."
                            )
                            default_score = None  # Reset if value isn't in options

                    teacher_score = st.radio(
                        "Score",
                        options=score_options_list,
                        index=score_index,
                        help="Select score from 4-1",
                        label_visibility="collapsed",
                        key=f"teacher_score_{selected_email}",  # Unique key per response
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
                        key=f"teacher_notes_{selected_email}",  # Unique key per response
                    )
                    st.caption(f"{len(teacher_notes)}/500 characters")

                # Navigation controls at bottom
                st.divider()
                nav_cols = st.columns(2)

                # Back button
                if current_idx > 0:
                    back_target = response_options[current_idx - 1]
                    nav_cols[0].button(
                        "← Back",
                        use_container_width=True,
                        on_click=set_selection_state,
                        kwargs={"target": back_target},  # Pass as kwargs
                    )

                # Next button
                if current_idx < len(response_options) - 1:
                    next_target = response_options[current_idx + 1]
                    nav_cols[1].button(
                        "Next →",
                        use_container_width=True,
                        on_click=set_selection_state,
                        kwargs={"target": next_target},  # Pass as kwargs
                    )

                # Run evaluation if requested (check rerun_button flag)
                if llm and evaluation_prompt and rerun_button:
                    with st.spinner("Running AI evaluation..."):
                        try:
                            st.session_state.confirm_rerun = (
                                False  # Reset confirm after trigger
                            )
                            logger.info(
                                f"Starting AI evaluation/rerun process for {selected_email}"
                            )
                            evaluation = run_evaluation(
                                llm,
                                evaluation_prompt,
                                selected_row.get(PART1_COL, ""),
                                selected_row.get(PART2_COL, ""),
                            )
                            logger.debug(
                                f"AI evaluation result for {selected_email}: {evaluation}"
                            )

                            # Save the evaluation result
                            if evaluation and isinstance(
                                evaluation, dict
                            ):  # Ensure valid result
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
                                    st.success("AI evaluation saved successfully!")
                                    # Update prev_eval in the current run so UI updates immediately if needed
                                    prev_eval = evaluation
                                    st.rerun()
                                else:
                                    st.error("Failed to save AI evaluation")
                            else:
                                st.error("AI evaluation returned no valid results")

                        except Exception as e:
                            st.error(f"Error running AI evaluation: {str(e)}")
                            logger.error(
                                f"AI evaluation error for {selected_email}: {str(e)}"
                            )
                            logger.error(
                                f"Selected row columns: {selected_row.index.tolist()}"
                            )

            except (IndexError, ValueError) as e:
                # Check if the error is specifically because the index wasn't found
                if isinstance(e, ValueError) and "is not in list" in str(e):
                    logger.warning(
                        f"Selected response '{current_response}' is not in the current 'response_options' list (likely due to filtering). Clearing selection."
                    )
                    st.warning(
                        "The previously selected response is no longer visible with the current filters. Please select another response."
                    )
                elif isinstance(e, IndexError):
                    logger.error(
                        f"IndexError accessing filtered_df for {selected_email}. Filtered list might be empty unexpectedly. {e}"
                    )
                    st.error(
                        "An internal error occurred trying to access the selected response data."
                    )
                else:  # Handle other potential ValueErrors like parsing
                    st.error(
                        f"Could not find or parse the selected response '{current_response}'. It might be invalid. Please select another response."
                    )
                    logger.error(f"Error loading selected response: {e}")

                # Clear invalid selection state regardless of specific error type
                if "selected_response" in st.session_state:
                    del st.session_state["selected_response"]
                current_response = None  # Clear local var
                st.query_params.clear()
                st.rerun()

        else:
            # Welcome/instruction state (shows on initial load or if no response selected)
            st.info(
                "👈 Find your students by selecting your name and class period in the sidebar, then choose a response."
            )

            # Add helpful instructions
            st.markdown(
                """
            ### 🦎 Green Anole Assessment Review Tool

            All responses are loaded. Three steps to review:

            #### 1. Find Students
            Select your name and class period in the sidebar, then choose a specific response.

            #### 2. Generate Feedback
            Click "🤖 Evaluate with AI" to analyze a response. You can also "🔄 Rerun Evaluation" if needed. Use "🤖 Evaluate X Responses" in the sidebar for batch processing.

            #### 3. Review & Finalize
            - Check AI feedback and scores.
            - Add your score (1-4) and notes (Optional).
            - Click "Done ✨" when finished with a response.
            - Results are saved automatically.

            """
            )

            # Show progress metrics if available
            if "df" in locals() and not df.empty:
                st.divider()
                st.markdown("### 📊 Overall Progress")

                # Calculate metrics
                total = len(df)
                evaluated = (
                    df["ai_evaluation"]
                    .apply(lambda x: isinstance(x, dict) and bool(x))
                    .sum()
                )
                finalized = df["finalized"].fillna(False).sum()
                in_progress = evaluated - finalized
                not_started = total - evaluated  # Simpler calculation

                # Display metrics in columns
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Responses", total)
                with col2:
                    st.metric(
                        "⚪️ Not Started",
                        f"{not_started}",
                    )
                with col3:
                    st.metric(
                        " 🟡 AI Evaluated",
                        f"{in_progress}",
                    )

                with col4:
                    st.metric(
                        " 🟢 Teacher Scored",
                        f"{finalized}",
                    )

            # Set the flag indicating the initial load process is complete
            st.session_state.initial_load_complete = True

        # Update filtered metrics for session state
        st.session_state.filtered_total = filtered_total
        st.session_state.filtered_evaluated = filtered_evaluated
        st.session_state.filtered_finalized = filtered_finalized
        st.session_state.filtered_in_progress = filtered_in_progress
        st.session_state.filtered_not_started = filtered_not_started

        # Store the current DataFrame in session state
        st.session_state.df = df

else:
    logger.error("Failed to load responses from Google Sheet")
    st.error(
        "🔴 Failed to load responses from Google Sheet. Please check logs or credentials."
    )


def get_current_time() -> str:
    """Get current time in ISO format."""
    return datetime.now().isoformat()
