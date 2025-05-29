import { Pool } from "pg";

const pool = new Pool({
  connectionString: process.env.POSTGRES_URL,
});

export async function listSamples() {
  const { rows } = await pool.query("SELECT * FROM anonymised_sample");
  // Map rows to match the expected shape if needed
  return rows.map((row) => ({
    sample_id: row.sample_id,
    pseudo_student_id: row.pseudo_student_id,
    text: row.text || row.content_redacted?.text || "",
    tags: row.tags || [],
  }));
}

export async function getFeedbackQueue(userId: string) {
  const { rows } = await pool.query("select * from get_feedback_queue($1)", [
    userId,
  ]);
  return rows.map((row) => ({
    feedback_id: row.feedback_id,
    submission_id: row.submission_id,
    student_name: row.student_name,
    assignment_title: row.assignment_title,
    draft: row.draft,
  }));
}

export async function publishFeedback(feedbackId: string) {
  await pool.query(
    "update narrative_feedback set feedback_status = 'published' where feedback_id = $1",
    [feedbackId],
  );
}
