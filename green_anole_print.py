# Configuration
DISPLAY_OPTIONS = {
    "show_teacher_score": True,  # Set to False to hide teacher scores
    "show_teacher_notes": True,  # Set to False to hide teacher notes
    "show_ai_feedback": True,  # Set to False to hide AI feedback
}

# Load required libraries
import pandas as pd
import numpy as np
from fpdf import FPDF, HTMLMixin
import markdown
from datetime import datetime
import re
import os


def clean_text(text):
    """Clean text of special characters that might cause issues."""
    if pd.isna(text):
        return ""

    # Convert to string
    text = str(text)

    # Replace problematic characters
    replacements = {
        "—": "-",  # em dash
        "–": "-",  # en dash
        """: '"',  # smart quotes
        """: '"',
        "'": "'",  # smart apostrophes
        "'": "'",
        "…": "...",  # ellipsis
        "\u2019": "'",  # right single quotation mark
        "\u2018": "'",  # left single quotation mark
        "\u201C": '"',  # left double quotation mark
        "\u201D": '"',  # right double quotation mark
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2026": "...",  # horizontal ellipsis
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove any other non-ASCII characters
    text = "".join(char if ord(char) < 128 else " " for char in text)

    return text


class StudentReportPDF(FPDF, HTMLMixin):
    def __init__(self, teacher_name="", hour=""):
        # Use landscape orientation with narrow margins for compact layout
        super().__init__(orientation="L", unit="mm", format="Letter")
        self.teacher_name = teacher_name
        self.hour = hour
        # Narrow margins and tighter page break
        self.set_margins(left=10, top=10, right=10)
        self.set_auto_page_break(auto=True, margin=10)

    def header(self):
        # Add header with title, teacher, hour and date
        self.set_font("Helvetica", "B", 15)
        self.cell(
            0,
            8,
            "Student Response Analysis",
            border=0,
            new_x="LMARGIN",
            new_y="NEXT",
        )
        self.set_font("Helvetica", "", 12)
        self.cell(
            0,
            6,
            f"{self.teacher_name} - {self.hour}",
            border=0,
            new_x="LMARGIN",
            new_y="NEXT",
        )
        self.set_font("Helvetica", "", 10)
        self.cell(
            0,
            4,
            datetime.now().strftime("%Y-%m-%d"),
            border=0,
            new_x="LMARGIN",
            new_y="NEXT",
        )
        self.ln(5)

    def footer(self):
        # Add page numbers
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(
            0,
            10,
            f"Page {self.page_no()}",
            border=0,
            new_x="RIGHT",
            new_y="TOP",
            align="C",
        )

    def add_section_title(self, title, size=12):
        # Smaller title font for compactness
        self.set_font("Helvetica", "B", size - 1)
        self.set_fill_color(240, 240, 240)  # Light gray background
        self.cell(
            0,
            6,
            clean_text(title),
            border=0,
            new_x="LMARGIN",
            new_y="NEXT",
            align="L",
            fill=True,
        )
        self.ln(1)

    def add_content(self, content, size=11):
        """Render markdown-formatted text content in the PDF."""
        # Clean text and convert markdown to HTML
        content_str = clean_text(content)
        html = markdown.markdown(content_str)
        # Tighter content spacing
        self.set_font("Helvetica", "", size - 1)
        self.write_html(html)
        self.ln(3)


def create_student_reports(df):
    # Create output directory if it doesn't exist
    output_dir = "student_reports"
    os.makedirs(output_dir, exist_ok=True)

    # Group the DataFrame by teacher and hour
    grouped = df.groupby(["Teacher Name", "Hour"])

    for (teacher_name, hour), group_df in grouped:
        # Create a sanitized filename
        safe_teacher_name = re.sub(r"[^\w\s-]", "", teacher_name).strip()
        output_file = os.path.join(
            output_dir, f"{safe_teacher_name}_Hour_{hour}_reports.pdf"
        )

        pdf = StudentReportPDF(teacher_name=teacher_name, hour=hour)

        for _, student_row in group_df.iterrows():
            pdf.add_page()

            # Student Name and Email
            student_name = student_row["Name"] if "Name" in student_row else "Unknown"
            pdf.add_section_title(f"Student: {student_name}", 14)
            pdf.add_content(f"Email: {student_row['Email Address']}")
            pdf.ln(5)

            # Annotated Response
            pdf.add_section_title("Annotated Response:")
            pdf.add_content(student_row["ai_annotated_response"])

            # Teacher Notes (if enabled and exists)
            if DISPLAY_OPTIONS["show_teacher_notes"]:
                teacher_notes = student_row["teacher_notes"]
                if pd.notna(teacher_notes) and str(teacher_notes).strip():
                    pdf.add_section_title(f"Notes from {teacher_name}:")
                    pdf.add_content(teacher_notes)

            # Teacher Score (if enabled and exists)
            if DISPLAY_OPTIONS["show_teacher_score"]:
                teacher_score = (
                    student_row["teacher_score"]
                    if "teacher_score" in df.columns
                    else None
                )
                if pd.notna(teacher_score):
                    pdf.add_section_title("Teacher Score:")
                    pdf.add_content(str(teacher_score))

            # AI Feedback (if enabled)
            if DISPLAY_OPTIONS["show_ai_feedback"]:
                pdf.add_section_title("AI Feedback:")

                # Strengths
                pdf.set_font("Helvetica", "B", 11)
                pdf.cell(
                    0,
                    7,
                    "Strengths:",
                    border=0,
                    new_x="LMARGIN",
                    new_y="NEXT",
                    align="L",
                )
                pdf.add_content(student_row["ai_feedback_strengths"])

                # Suggestions
                pdf.set_font("Helvetica", "B", 11)
                pdf.cell(
                    0,
                    7,
                    "Suggestions for Improvement:",
                    border=0,
                    new_x="LMARGIN",
                    new_y="NEXT",
                    align="L",
                )
                pdf.add_content(student_row["ai_feedback_suggestions"])

        # Save the document
        pdf.output(output_file)
        print(f"PDF report has been generated: {output_file}")


# Load the CSV file into a pandas DataFrame
df = pd.read_csv("data/green_anole/green_anole_data.csv")

# Create the reports
create_student_reports(df)

# Print summary information
print("\nSummary of reports generated:")
grouped_summary = df.groupby(["Teacher Name", "Hour"]).size()
for (teacher, hour), count in grouped_summary.items():
    print(f"{teacher} - {hour}: {count} students")
