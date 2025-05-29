import { Pool } from 'pg';

const pool = new Pool({
  connectionString: process.env.POSTGRES_URL,
});

export async function listSamples() {
  const { rows } = await pool.query('SELECT * FROM anonymised_sample');
  // Map rows to match the expected shape if needed
  return rows.map(row => ({
    sample_id: row.sample_id,
    pseudo_student_id: row.pseudo_student_id,
    text: row.text || row.content_redacted?.text || '',
    tags: row.tags || [],
  }));
}
