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
from datetime import datetime
import uuid
from sqlalchemy import Float

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

client_config = st.secrets.get("client", {})
logger.debug(f"client_config: {client_config}")

st.session_state.allowed_emails = client_config.get("allowed_emails", "").split(",")
st.session_state.email_to_teacher = client_config.get("email_teacher_name", None)

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


@st.cache_data(ttl=900)  # 15-minute cache
def load_gsheet_df() -> pd.DataFrame:
    """Load and preprocess Google Sheet data with column detection.
    Returns every 15 minutes for each user session."""
    df = get_google_sheet()
    if df is not None:
        return detect_and_rename(df)
    return pd.DataFrame()  # Return empty DataFrame if load fails


def generate_response_options(
    df: pd.DataFrame, email_col: str, name_col: str, key_suffix: str = ""
) -> list[str]:
    """Generate a list of response options with status indicators.

    Args:
        df (pd.DataFrame): DataFrame containing response data
        email_col (str): Name of the email column
        name_col (str): Name of the student name column
        key_suffix (str): Optional suffix to make checkbox key unique

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

    # Initialize or get the alias mapping
    if "alias_mapping" not in st.session_state:
        st.session_state.alias_mapping = {}

    # Get show_names value from session state
    show_names = st.session_state.get("show_names", False)
    logger.debug(f"Generating options with show_names={show_names}")

    # Sort DataFrame based on show_names setting
    if show_names:
        df_sorted = df.sort_values(by=[name_col, email_col])
    else:
        df_sorted = df.sort_values(by=[email_col])

    # Create aliases using DataFrame index (or use a stable ID if available)
    # Let's stick with index for alias generation for now
    for idx, row in df_sorted.iterrows():
        email = row[email_col]
        if email not in st.session_state.alias_mapping:
            # Use the original DataFrame's index for a potentially more stable ID
            # Requires passing the original df or finding the row there.
            # Simpler: Use the index from the *sorted* df for display ID.
            st.session_state.alias_mapping[email] = f"Response #{idx+1}"

    email_options = []
    display_mapping = {}

    for idx, row in df_sorted.iterrows():
        email = row[email_col]
        name = row.get(name_col, "Unknown Name")

        # Determine status indicator based on new criteria
        is_teacher_reviewed = pd.notna(row.get("teacher_score")) or (
            row.get("teacher_notes") and str(row.get("teacher_notes")).strip()
        )
        has_ai_evaluation = isinstance(row.get("ai_evaluation"), dict) and row.get(
            "ai_evaluation"
        )

        if is_teacher_reviewed:
            status = "🟢"  # Done (Teacher scored or noted)
        elif has_ai_evaluation:
            status = "🟡"  # In progress (AI eval done, but no teacher input)
        else:
            status = "⚪️"  # Not started

        # Generate display text based on show_names
        response_display_id = f"#{idx+1}"  # Use sorted index for display ID
        if show_names:
            display_text = f"{status} {response_display_id} {name} (`{email}`)"
        else:
            display_text = f"{status} Response {response_display_id}"

        email_options.append(email)
        display_mapping[email] = display_text

    return email_options, display_mapping


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


# Define the Feedback model for tracking teacher actions
class Feedback(Base):
    __tablename__ = "feedback"

    # Fields from the spec
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=datetime.utcnow)
    modified_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    session_id = Column(
        String, index=True
    )  # e.g. your app session or LangSmith session
    run_id = Column(String, index=True)  # e.g. the LLM run/span ID, if available
    key = Column(String, nullable=False)  # e.g. "teacher_score", "ai_rerun", "clear"
    score = Column(Float, nullable=True)  # numerical score
    value = Column(String, nullable=True)  # categorical value, if any
    comment = Column(String, nullable=True)  # teacher's notes
    correction = Column(JSON, nullable=True)  # any structured correction object
    # feedback_source object flattened:
    feedback_source_type = Column(String, nullable=False)
    feedback_source_metadata = Column(JSON, nullable=True)
    feedback_source_user_id = Column(String, nullable=False)
    ai_evaluation = Column(
        JSON, nullable=True
    )  # Store the AI evaluation at the time of feedback


def log_feedback(
    session_id: str,
    run_id: str,
    key: str,
    score: float | None = None,
    value: str | None = None,
    comment: str | None = None,
    correction: dict | None = None,
    ai_evaluation: dict | None = None,  # Added parameter
    source_type: str = "app",
    source_metadata: dict | None = None,
):
    """Insert a feedback record per LangSmith's spec."""
    db = SessionLocal()
    try:
        db.add(
            Feedback(
                session_id=session_id,
                run_id=run_id,
                key=key,
                score=score,
                value=value,
                comment=comment,
                correction=correction,
                ai_evaluation=ai_evaluation,  # Added field
                feedback_source_type=source_type,
                feedback_source_metadata=source_metadata,
                feedback_source_user_id=st.experimental_user.email,
            )
        )
        db.commit()
        logger.info(f"Successfully logged feedback: {key}")
    except Exception as e:
        logger.error(f"Failed to log feedback {key}: {e}")
        db.rollback()
    finally:
        db.close()


# Create the tables if they don't exist
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


# --- Helper Function for Column Detection ---
def find_column(df_columns, potential_names):
    """Find the first matching column name from a list of potential names."""
    for name in potential_names:
        if name in df_columns:
            logger.info(f"Detected column: '{name}'")
            return name
    logger.warning(f"Could not find any of the potential columns: {potential_names}")
    return None


def get_current_time() -> str:
    """Get current time in ISO format."""
    return datetime.now().isoformat()


def build_filter_widgets(
    df: pd.DataFrame, teacher_col: str | None, hour_col: str | None
) -> None:
    """Build all sidebar filter widgets outside of fragments.

    Args:
        df: The DataFrame containing response data
        teacher_col: Name of the teacher column, if it exists
        hour_col: Name of the hour column, if it exists
    """
    with st.sidebar:
        st.header("🔍 Select Responses")

        # Teacher filter
        if teacher_col:
            teachers: list[str] = ["All"] + sorted(df[teacher_col].unique().tolist())
            # Determine the index for the teacher selectbox based on session state
            teacher_index = 0  # Default to "All"
            selected_teacher = st.session_state.get("selected_teacher", "All")
            if selected_teacher in teachers:
                try:
                    teacher_index = teachers.index(selected_teacher)
                except ValueError:
                    logger.warning(
                        f"Saved teacher '{selected_teacher}' not found in options. Defaulting to 'All'."
                    )
                    teacher_index = 0  # Fallback to "All" if saved teacher not in list

            st.selectbox(
                "Teacher:",
                teachers,
                key="selected_teacher",
                index=teacher_index,  # Use calculated index
            )

        # Hour filter
        if hour_col:
            hours: list[str] = ["All"] + sorted(df[hour_col].unique().tolist())
            st.selectbox(
                "Hour:",
                hours,
                key="selected_hour",
                index=0,
            )

        # Status filter
        status_options: list[str] = ["All", "Not Started", "In Progress", "Finalized"]
        st.selectbox(
            "Review Status:",
            status_options,
            key="selected_status",
            index=0,
        )

        # Add metrics section
        st.divider()


def apply_filters(
    df: pd.DataFrame, teacher_col: str | None, hour_col: str | None
) -> pd.DataFrame:
    """Apply filters to the DataFrame based on session state values."""
    filtered_df = df.copy()

    # Get filter values from session state
    selected_teacher = st.session_state.get("selected_teacher", "All")
    selected_hour = st.session_state.get("selected_hour", "All")
    selected_status = st.session_state.get("selected_status", "All")

    # Apply teacher filter
    if selected_teacher != "All" and teacher_col and teacher_col in filtered_df.columns:
        filtered_df = filtered_df[filtered_df[teacher_col] == selected_teacher]

    # Apply hour filter
    if selected_hour != "All" and hour_col and hour_col in filtered_df.columns:
        filtered_df = filtered_df[filtered_df[hour_col] == selected_hour]

    # Apply status filter
    if selected_status != "All":
        if selected_status == "Not Started":
            # Ensure necessary columns exist before filtering
            if (
                "ai_evaluation" in filtered_df.columns
                and "finalized" in filtered_df.columns
            ):
                filtered_df = filtered_df[
                    filtered_df["ai_evaluation"].apply(
                        lambda x: not isinstance(x, dict) or not x
                    )
                    & (~filtered_df["finalized"].fillna(False))
                ]
        elif selected_status == "In Progress":
            if (
                "ai_evaluation" in filtered_df.columns
                and "finalized" in filtered_df.columns
            ):
                filtered_df = filtered_df[
                    filtered_df["ai_evaluation"].apply(
                        lambda x: isinstance(x, dict) and bool(x)
                    )
                    & (~filtered_df["finalized"].fillna(False))
                ]
        elif selected_status == "Finalized":
            if "finalized" in filtered_df.columns:
                filtered_df = filtered_df[filtered_df["finalized"].fillna(False)]

    return filtered_df


def get_google_sheet():
    """Connect to Google Sheets and return the worksheet data with normalized column names"""

    # Assert authorized user
    try:
        email = st.experimental_user.email
        assert email in st.session_state.get(
            "allowed_emails", []
        ), "You are not authorized to access this tool."
    except Exception as e:
        logger.error(f"Error getting user email: {e}")
        st.error(
            "You are not authorized to access this tool. Contact aw@eddolearning.com for access."
        )
        logger.error(
            f"User {st.experimental_user.email} is not authorized to access this tool."
            f"Allowed emails: {st.session_state.get('allowed_emails', [])}"
        )
        st.stop()

    try:
        # Debug: Print credentials path
        logger.debug("Attempting to load credentials...")
        creds_dict = st.secrets["google_service_account"]
        # Use Credentials.from_service_account_info to parse the dictionary
        creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPE)
        # Create a client to interact with Google Sheets using the parsed credentials
        client = gspread.authorize(creds)
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
            # Do NOT log sample values!

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
        total_to_evaluate = len(unevaluated_df)
        logger.info(f"Found {total_to_evaluate} responses for batch evaluation.")

        # Create a progress bar
        progress_bar = st.progress(0)
        status_container = st.empty()
        results_container = st.empty()
        results_text = []

        evaluation_count = 0
        success_count = 0
        error_count = 0

        # Process in batches
        for start in range(0, len(unevaluated_df), batch_size):
            batch = unevaluated_df.iloc[start : start + batch_size]

            for idx, row in batch.iterrows():
                try:
                    email = row[email_col]
                    name = row.get("name", "Unknown")  # Get student name if available
                    status_container.write(
                        f"🔄 Evaluating response from {name} ({email})"
                    )

                    # Run evaluation
                    evaluation = run_evaluation(
                        llm,
                        evaluation_prompt,
                        row[PART1_COL],
                        row[PART2_COL],
                    )

                    if evaluation:
                        # Save results using the database function
                        save_success = save_enriched_data(
                            email=email,
                            ai_output=evaluation,
                            teacher_notes=row.get("teacher_notes", ""),
                            teacher_score=row.get("teacher_score", None),
                            finalized=row.get("finalized", False),
                        )
                        if save_success:
                            success_count += 1
                            overall_score = evaluation.get("rubric_scores", {}).get(
                                "Overall", "N/A"
                            )
                            results_text.append(
                                f"✅ {name} ({email}): Score {overall_score}/4"
                            )
                            # Update the DataFrame in memory
                            df.loc[df[email_col] == email, "ai_evaluation"] = [
                                evaluation
                            ]
                            df.loc[df[email_col] == email, "last_modified"] = (
                                datetime.now()
                            )
                        else:
                            error_count += 1
                            results_text.append(
                                f"❌ {name} ({email}): Failed to save evaluation"
                            )
                    else:
                        error_count += 1
                        results_text.append(
                            f"❌ {name} ({email}): No evaluation results"
                        )

                except Exception as e:
                    error_count += 1
                    results_text.append(f"❌ {name} ({email}): Error - {str(e)}")
                    logger.error(f"Error evaluating response for {email}: {e}")

                evaluation_count += 1
                # Update progress
                progress = evaluation_count / total_to_evaluate
                progress_bar.progress(progress)
                # Show running results, keeping last 10 visible
                results_container.markdown(
                    "**Recent Results:**\n" + "\n".join(results_text[-10:])
                )

        # Final status update
        status_container.write(
            f"✨ Batch evaluation complete! "
            f"Processed {evaluation_count} responses: "
            f"{success_count} successful, {error_count} errors"
        )
        # Show all results in an expander
        with st.expander("View All Results"):
            st.markdown("\n".join(results_text))

        logger.info(
            f"Batch evaluation complete. Evaluated {evaluation_count} responses."
        )
        return df

    except Exception as e:
        logger.error(f"Batch evaluation failed: {e}")
        return df


# --- Streamlit App ---
st.title("🦎 Green Anole Assessment Review with AI Assist")

# --- Session State Initialization ---
if "initial_load_complete" not in st.session_state:
    st.session_state.initial_load_complete = False
    st.session_state.initial_teacher_set = False  # Flag for initial teacher setting
if "selected_response" not in st.session_state:
    st.session_state.selected_response = None  # Initialize if not present
if "confirm_rerun" not in st.session_state:
    st.session_state.confirm_rerun = False  # Initialize confirmation flag
if "app_session_id" not in st.session_state:
    st.session_state.app_session_id = str(uuid.uuid4())  # Generate unique session ID


# --- Callback Functions ---
def set_selection_state(target=None):
    """Callback to update selection state. Gets value directly from session state."""
    logger.debug(f"Callback set_selection_state called with target: {target}")
    if target is not None:
        st.session_state.response_selector = target
    elif "response_selector" not in st.session_state:
        # Initialize with first option if available
        if "response_options" in st.session_state and st.session_state.response_options:
            st.session_state.response_selector = st.session_state.response_options[0]
        else:
            st.session_state.response_selector = None

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
        return True  # Indicate success
    else:
        st.error("Failed to save finalization.")
        return False  # Indicate failure


def clear_teacher_evaluation(email: str):
    """Callback to clear teacher score and notes while preserving AI evaluation.

    Args:
        email (str): The email identifier for the response to clear.
    """
    if not email:
        logger.error("Clear teacher evaluation called without a valid email.")
        st.error("Could not identify the response to clear.")
        return

    logger.info(f"Attempting to clear teacher evaluation for {email}")
    # Get the current AI evaluation before clearing
    db = SessionLocal()
    current_ai_eval = None
    save_success = False
    try:
        existing_record = (
            db.query(EnrichedData).filter(EnrichedData.Timestamp == email).first()
        )
        current_ai_eval = existing_record.ai_evaluation if existing_record else None
    except Exception as e:
        logger.error(f"Error fetching existing data for {email} before clear: {e}")
        st.error("Failed to load existing data before clearing.")
        db.close()
        return  # Stop if we can't fetch existing data
    finally:
        db.close()  # Ensure db is closed even on fetch error

    # Save the cleared state to database while preserving AI evaluation
    save_success = save_enriched_data(
        email=email,
        ai_output=current_ai_eval,  # Preserve existing AI evaluation
        teacher_notes="",  # Clear notes
        teacher_score=None,  # Clear score
        finalized=False,  # Unfinalize when clearing
    )

    # Log feedback if save was successful
    if save_success:
        log_feedback(
            session_id=st.session_state.get("app_session_id", "local"),
            run_id=email,  # Using email as run_id
            key="clear",
            source_type="app",
            source_metadata={
                "previous_ai_eval": bool(current_ai_eval)
            },  # Track if AI eval was preserved
            ai_evaluation=current_ai_eval,  # Pass the AI evaluation
        )

    # --- Update DataFrame in memory ---
    if save_success and "df" in st.session_state:
        try:
            df_current = st.session_state.df
            # Need email_col to find the row
            email_col = "email"  # Assuming this is the correct column name
            idx = df_current.index[df_current[email_col] == email].tolist()
            if idx:
                update_idx = idx[0]
                df_current.loc[update_idx, "teacher_notes"] = ""
                df_current.loc[update_idx, "teacher_score"] = None
                df_current.loc[update_idx, "finalized"] = (
                    False  # Also unfinalize in memory
                )
                df_current.loc[update_idx, "last_modified"] = datetime.now()
                st.session_state.df = df_current  # Store updated df back
                logger.debug(f"Updated session state df after clearing for {email}")
            else:
                logger.error(
                    f"Could not find index for {email} in session_state df after clear."
                )
        except Exception as e:
            logger.error(f"Error updating session state df after clear: {e}")
    elif not save_success:
        logger.error(
            f"Skipping session state df update for {email} due to save failure."
        )
    elif "df" not in st.session_state:
        logger.error(
            f"Skipping session state df update for {email} because 'df' not in session state."
        )

    # Delete the specific keys from session state to reset widgets
    score_key = f"teacher_score_{email}"
    notes_key = f"teacher_notes_{email}"
    if score_key in st.session_state:
        del st.session_state[score_key]
    if notes_key in st.session_state:
        del st.session_state[notes_key]

    # 🔄 refresh the full app now, fragment scope won't work
    st.rerun()


@st.dialog("Confirm Clear")
def confirm_clear_dialog(email_to_clear):
    """Displays a confirmation dialog for clearing teacher scores and notes."""
    st.warning(
        f"Are you sure you want to clear the score and notes for {email_to_clear}?"
    )
    col1, col2 = st.columns(2)  # Use st.columns inside the dialog function
    with col1:
        if st.button("Yes, Clear", type="primary", use_container_width=True):
            clear_teacher_evaluation(email_to_clear)  # Pass email
            # The rerun in clear_teacher_evaluation will close the dialog
    with col2:
        if st.button("Cancel", use_container_width=True):
            # No action needed, simply closes the dialog by ending the function run
            pass


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

    # --- Set Initial Teacher Filter (Only Once Per Session) ---
    if (
        not st.session_state.get("initial_teacher_set", False)
        and teacher_col in df_raw.columns
    ):
        try:
            user_email = st.experimental_user.email
            if "EMAIL_TO_TEACHER" in st.secrets:
                email_to_teacher = st.secrets["EMAIL_TO_TEACHER"]
                default_teacher = email_to_teacher.get(
                    user_email
                )  # Returns None if not found
                if default_teacher and default_teacher in df_raw[teacher_col].unique():
                    st.session_state.selected_teacher = default_teacher
                    logger.info(
                        f"Setting initial teacher filter to '{default_teacher}' for user '{user_email}'."
                    )
                else:
                    if default_teacher:
                        logger.warning(
                            f"Teacher '{default_teacher}' mapped to user '{user_email}' not found in sheet. Defaulting to 'All'."
                        )
                    st.session_state.selected_teacher = (
                        "All"  # Default if not found or not in sheet
                    )
            else:
                logger.warning(
                    "'EMAIL_TO_TEACHER' not found in secrets. Defaulting teacher filter to 'All'."
                )
                st.session_state.selected_teacher = "All"  # Default if secrets missing

            st.session_state.initial_teacher_set = True  # Mark as set
        except Exception as e:
            logger.error(f"Error setting initial teacher filter: {e}")
            st.session_state.selected_teacher = "All"  # Default on any error
            st.session_state.initial_teacher_set = True  # Prevent retrying on error

    if df_raw.empty:
        logger.warning("No responses found!")
    else:
        # Merge with enriched data - pass detected email_col
        df = get_merged_data(df_raw, email_col)
        logger.info(f"Loaded {len(df)} responses successfully!")

        # Build filter widgets before applying filters
        build_filter_widgets(df, teacher_col, hour_col)

        # Apply filters based on session state
        df_filtered = apply_filters(df, teacher_col, hour_col)

        # --- State Synchronization (Session State & Query Params) ---
        # 1. Get potential values from session state and query params
        state_response = st.session_state.get("selected_response", None)
        qp_response = st.query_params.get("response", None)

        # 2. Generate response options first so we can validate against them
        response_options, display_mapping = generate_response_options(
            df_filtered, email_col, name_col, "_main"
        )

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
        # build_filter_widgets(df, teacher_col, hour_col) # REMOVE THIS LINE

        # Generate response options from filtered DataFrame
        response_options, display_mapping = generate_response_options(
            df_filtered, email_col, name_col, "_main"
        )

        # Store in session state for consistency
        st.session_state.response_options = response_options

        if logger.getEffectiveLevel() <= logging.DEBUG:
            logger.debug(f"Filtered DataFrame Length: {len(df)}")

        # Progress metrics for filtered data
        st.sidebar.divider()

        # Calculate metrics for filtered data
        filtered_total = len(df_filtered)
        # --- Recalculate counts based on status icon logic --- #
        filtered_teacher_scored = df_filtered[
            pd.notna(df_filtered["teacher_score"])
            | (df_filtered["teacher_notes"].fillna("").str.strip() != "")
        ].shape[0]
        filtered_ai_evaluated_only = df_filtered[
            df_filtered["ai_evaluation"].apply(
                lambda x: isinstance(x, dict) and bool(x)
            )
            & ~(  # Exclude those already teacher scored
                pd.notna(df_filtered["teacher_score"])
                | (df_filtered["teacher_notes"].fillna("").str.strip() != "")
            )
        ].shape[0]
        filtered_not_started = (
            filtered_total - filtered_teacher_scored - filtered_ai_evaluated_only
        )
        # --- End Recalculation --- #

        # Combined section for filter summary and response selection
        st.sidebar.markdown(
            f"""
        ### 📝 Responses
        {filtered_total} total • ⚪️ {filtered_not_started} • 🟡 {filtered_ai_evaluated_only} • 🟢 {filtered_teacher_scored}
        """
        )

        # --- MOVED BATCH EVALUATION BUTTON HERE --- #
        # Add batch evaluation callback
        def run_batch_evaluation():
            """Callback to run batch evaluation and ensure UI updates."""
            if not llm or not evaluation_prompt:
                st.error("AI evaluation is not available.")
                return

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
                f"🤖 Evaluate {filtered_not_started} New Responses",
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
        # --- END MOVED BATCH EVALUATION BUTTON --- #

        # Add show names checkbox just before the response list
        def on_show_names_change():
            """Callback when show_names checkbox changes."""
            # Force rerun to update all displays
            pass

        # Place the checkbox with a static key and explicit callback
        show_names = st.sidebar.checkbox(
            "Show Names",
            value=st.session_state.get("show_names", False),
            key="show_names",
            on_change=on_show_names_change,
        )

        # Find the index of the current selection in the options list
        current_selection_index = None
        if current_response and response_options:
            try:
                current_selection_index = response_options.index(current_response)
            except ValueError:
                # If current selection is not in filtered list, clear it
                if "selected_response" in st.session_state:
                    del st.session_state["selected_response"]
                current_response = None
                st.query_params.clear()
                current_selection_index = 0

        # Add the radio list to choose a response (directly in sidebar)
        st.sidebar.radio(
            "Select Response:",
            options=response_options,  # List of emails
            index=current_selection_index if current_selection_index is not None else 0,
            format_func=lambda email: display_mapping.get(
                email, email
            ),  # Use mapping for display
            on_change=set_selection_state,  # Use the callback
            key="response_selector",  # Use a STATIC key
            label_visibility="collapsed",  # Hide label if desired
        )

        # --- Add Download Button for Filtered Data ---
        if not df_filtered.empty:
            # Prepare data for download (convert dicts/lists in ai_evaluation to strings if needed)
            df_download = df_filtered.copy()
            if "ai_evaluation" in df_download.columns:
                df_download["ai_evaluation"] = df_download["ai_evaluation"].apply(
                    lambda x: str(x) if isinstance(x, (dict, list)) else x
                )

            csv = df_download.to_csv(index=False).encode("utf-8")
            st.sidebar.download_button(
                label=f"💾 Download {filtered_total} Selected Responses (.csv)",
                data=csv,
                file_name=f"filtered_responses_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_filtered_responses",
                use_container_width=True,
            )
        # --- End Download Button ---

        # Main content area
        # Show response details only if a response is selected AND initial load is complete
        if current_response and st.session_state.get("initial_load_complete", False):
            try:
                # Find the selected row in the *original* DataFrame (df) using the email
                selected_rows = df[df[email_col] == current_response]

                if selected_rows.empty:
                    # Handle case where selected response is no longer in the main df
                    # (Should be less likely now, but good to keep)
                    logger.error(
                        f"Selected email '{current_response}' not found in main data. Clearing selection."
                    )
                    st.warning(
                        "The selected response data could not be found. Please select another response."
                    )
                    if "selected_response" in st.session_state:
                        del st.session_state["selected_response"]
                    current_response = None
                    st.query_params.clear()
                    st.rerun()  # Rerun to clear the view

                # Proceed assuming selected_rows is not empty
                selected_row = selected_rows.iloc[0]
                # Find index in the *email* options list (for nav buttons)
                # Ensure current_response (email) is still valid before finding index
                if current_response in response_options:
                    current_idx = response_options.index(current_response)
                else:
                    # This case means the selected email is valid in `df` but not in the current `response_options` (e.g., due to filters)
                    logger.warning(
                        f"Selected email '{current_response}' exists in data but not in current filter options. Clearing selection."
                    )
                    st.warning(
                        "The previously selected response is not visible with the current filters. Please select another response."
                    )
                    if "selected_response" in st.session_state:
                        del st.session_state["selected_response"]
                    current_response = None
                    st.query_params.clear()
                    st.rerun()

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
                        st.warning("🟡 Ready for Review")
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
                                f"AI evaluation for {current_response} was a string, attempting parse."
                            )
                            prev_eval = json.loads(evaluation_data)
                        else:
                            # This case should be less likely now with the improved has_evaluation check
                            logger.warning(
                                f"AI evaluation data for {current_response} is not a valid dict: {evaluation_data}"
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
                                f"Evaluation data existed for {current_response} but failed to load as dict: {evaluation_data}"
                            )
                            st.error(
                                "Could not load previous evaluation data despite it being present."
                            )

                    except json.JSONDecodeError as e:
                        st.error(
                            f"Could not parse previous evaluation data (JSON error): {e}"
                        )
                        logger.error(
                            f"Error parsing evaluation JSON for {current_response}: {e}"
                        )
                    except Exception as e:
                        st.error(f"Could not load previous evaluation data: {e}")
                        logger.error(
                            f"Error loading evaluation data for {current_response}: {e}"
                        )

                # Display AI scores just before teacher evaluation
                # Check if prev_eval is a valid dictionary before proceeding
                rerun_button = False  # Initialize rerun_button flag

                st.markdown("#### ✨🦔 AI Evaluation")

                if not prev_eval or not isinstance(prev_eval, dict):
                    # Show evaluate button when no evaluation exists
                    if st.button(
                        "Evaluate with AI",
                        icon="✨",
                        type="primary",
                        use_container_width=True,
                        key="eval_button",
                    ):
                        # DEBUG: Log input to evaluation
                        logger.debug(
                            f"[EVAL] Triggered for email={current_response}, part1={selected_row.get(PART1_COL, '')[:40]}, part2={selected_row.get(PART2_COL, '')[:40]}"
                        )
                        st.session_state.confirm_rerun = False  # Reset just in case
                        rerun_button = True  # Set flag to proceed
                        # Store current selection before rerunning
                        st.session_state.selected_response = current_response
                        # Actually run the evaluation (simulate what happens on rerun)
                        try:
                            eval_result = run_evaluation(
                                llm,
                                evaluation_prompt,
                                selected_row.get(PART1_COL, ""),
                                selected_row.get(PART2_COL, ""),
                            )
                            # DEBUG: Log output type and keys only
                            logger.debug(
                                f"[EVAL] Result type: {type(eval_result)}, keys: {list(eval_result.keys()) if isinstance(eval_result, dict) else 'N/A'}"
                            )
                        except Exception as e:
                            logger.error(f"[EVAL] Exception during evaluation: {e}")
                else:
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
                            rerun_button_clicked = (  # Use different variable name
                                st.button(
                                    "🔄🦔 Rerun Evaluation",
                                    type="secondary",
                                    use_container_width=True,
                                    key="eval_rerun_button",  # Keep key simple
                                )
                            )
                            # Add confirmation only for rerun
                            if rerun_button_clicked and has_evaluation:
                                if not st.session_state.get("confirm_rerun", False):
                                    st.session_state.confirm_rerun = True
                                    # Now just let the script continue to show the checkbox

                                if st.session_state.get("confirm_rerun", False):
                                    # Log the rerun attempt before proceeding
                                    log_feedback(
                                        session_id=st.session_state.get(
                                            "app_session_id", "local"
                                        ),
                                        run_id=current_response,  # Using email as run_id
                                        key="ai_rerun",
                                        source_type="app",
                                        source_metadata={
                                            "previous_scores": prev_eval.get(
                                                "rubric_scores", {}
                                            ),
                                        },
                                        ai_evaluation=prev_eval,  # Pass the previous AI evaluation
                                    )
                                    confirm_key = f"rerun_confirm_{current_response}"  # Unique key

                    except Exception as e:
                        st.error("Could not display AI scores")
                        logger.error(
                            f"Error displaying AI scores for {current_response}: {e}"
                        )

                # Teacher evaluation section
                st.markdown("### 👩‍🏫 Teacher Score")

                # Clear button triggers dialog
                if st.button(
                    "🗑️ Clear Score and Notes",
                    type="secondary",
                    help="Clear teacher score and notes",
                    key=f"clear_btn_{current_response}",  # Unique key per response
                ):
                    # Call the decorated dialog function
                    confirm_clear_dialog(current_response)  # Pass the email

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
                        key=f"teacher_score_{current_response}",  # Unique key per response
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
                        key=f"teacher_notes_{current_response}",  # Unique key per response
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

                    def save_if_changed_and_next(
                        current_email,
                        next_email,
                        current_notes_widget_key,
                        current_score_widget_key,
                        current_row_data,  # Pass the row data dictionary
                    ):
                        """Saves teacher input if changed, then navigates to the next response."""
                        logger.debug(
                            f"save_if_changed_and_next called for {current_email}, moving to {next_email}"
                        )

                        # Get current widget values
                        current_notes = st.session_state.get(
                            current_notes_widget_key, ""
                        )
                        current_score = st.session_state.get(
                            current_score_widget_key, None
                        )

                        # Get original values (or defaults if not present)
                        original_notes = current_row_data.get("teacher_notes", "")
                        original_score = current_row_data.get("teacher_score")
                        # Handle potential NaN from pandas for score comparison
                        original_score_int = None
                        if pd.notna(original_score):
                            try:
                                original_score_int = int(float(original_score))
                            except (ValueError, TypeError):
                                original_score_int = (
                                    None  # Treat invalid original scores as None
                                )

                        # Check if anything changed
                        notes_changed = current_notes != original_notes
                        score_changed = current_score != original_score_int

                        if notes_changed or score_changed:
                            logger.info(
                                f"Saving changes for {current_email} before moving to next."
                            )
                            save_success = save_enriched_data(
                                email=current_email,
                                ai_output=current_row_data.get(
                                    "ai_evaluation"
                                ),  # Preserve AI eval
                                teacher_notes=current_notes,  # Use widget value
                                teacher_score=current_score,  # Use widget value
                                finalized=current_row_data.get(
                                    "finalized", False
                                ),  # Preserve finalized status
                            )
                            if not save_success:
                                st.error(
                                    f"Failed to save changes for {current_email}. Please check logs."
                                )
                                logger.error(
                                    f"Failed to save changes automatically for {current_email} when navigating."
                                )
                            else:
                                # Log feedback after successful save
                                log_feedback(
                                    session_id=st.session_state.get(
                                        "app_session_id", "local"
                                    ),
                                    run_id=current_email,  # Using email as run_id since we don't track LLM runs
                                    key="teacher_score",
                                    score=(
                                        float(current_score)
                                        if current_score is not None
                                        else None
                                    ),
                                    comment=current_notes,
                                    source_type="app",
                                    ai_evaluation=current_row_data.get(
                                        "ai_evaluation"
                                    ),  # Pass the current AI evaluation
                                )
                                # --- Still proceed to next even if save fails ---
                        else:
                            logger.debug(
                                f"No changes detected for {current_email}, moving to next without saving."
                            )

                        # Navigate to the next response by setting state and rerunning
                        set_selection_state(target=next_email)
                        # No explicit rerun needed here, set_selection_state causes one if value changed

                    nav_cols[1].button(
                        "Next →",
                        use_container_width=True,
                        on_click=save_if_changed_and_next,
                        kwargs={
                            "current_email": current_response,
                            "next_email": next_target,
                            "current_notes_widget_key": f"teacher_notes_{current_response}",
                            "current_score_widget_key": f"teacher_score_{current_response}",
                            "current_row_data": selected_row.to_dict(),  # Pass current data
                        },
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
                        f"IndexError accessing filtered_df for {current_response}. Filtered list might be empty unexpectedly. {e}"
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
                # --- Recalculate counts based on status icon logic --- #
                teacher_scored = df[
                    pd.notna(df["teacher_score"])
                    | (df["teacher_notes"].fillna("").str.strip() != "")
                ].shape[0]
                ai_evaluated_only = df[
                    df["ai_evaluation"].apply(lambda x: isinstance(x, dict) and bool(x))
                    & ~(  # Exclude those already teacher scored
                        pd.notna(df["teacher_score"])
                        | (df["teacher_notes"].fillna("").str.strip() != "")
                    )
                ].shape[0]
                not_started = total - teacher_scored - ai_evaluated_only
                # --- End Recalculation --- #

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
                        "🟡 AI Evaluated",
                        f"{ai_evaluated_only}",
                    )

                with col4:
                    st.metric(
                        "🟢 Teacher Scored",
                        f"{teacher_scored}",
                    )

                # Add Start Review button after metrics
                st.divider()
                start_col1, start_col2, start_col3 = st.columns([1, 2, 1])
                with start_col2:
                    if response_options:  # Only show if there are responses to review
                        # Filter for responses that haven't been scored by teacher WITHIN THE FILTERED VIEW
                        unscored_df = df_filtered[
                            pd.isna(df_filtered["teacher_score"])
                        ]  # Use df_filtered here
                        if not unscored_df.empty:
                            # Generate options only for unscored responses FROM THE FILTERED VIEW
                            unscored_options, unscored_display_mapping = (
                                generate_response_options(
                                    unscored_df, email_col, name_col, "_unscored"
                                )
                            )
                            if unscored_options:
                                first_unscored = unscored_options[0]
                                st.button(
                                    # Use the length of the FILTERED unscored options
                                    f"✨ Review {len(unscored_options)} Selected Unscored Responses",
                                    type="primary",
                                    on_click=set_selection_state,
                                    kwargs={
                                        "target": first_unscored
                                    },  # Target the first from the FILTERED list
                                    use_container_width=True,
                                )
                            else:
                                st.success(
                                    "🎉 All filtered responses have been scored!"
                                )  # Adjusted message
                        else:
                            st.success(
                                "🎉 All filtered responses have been scored!"
                            )  # Adjusted message

            # Set the flag indicating the initial load process is complete
            st.session_state.initial_load_complete = True

        # Update filtered metrics for session state
        st.session_state.filtered_total = filtered_total
        st.session_state.filtered_not_started = filtered_not_started

        # Store the current DataFrame in session state
        st.session_state.df = df

else:
    logger.error("Failed to load responses from Google Sheet")
    st.error(
        "🔴 Failed to load responses from Google Sheet. Please check logs or credentials."
    )
