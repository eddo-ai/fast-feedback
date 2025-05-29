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

export async function getSampleDetail(sampleId: string) {
  const { rows } = await pool.query(
    `SELECT a.sample_id,
            a.pseudo_student_id,
            a.derived_at,
            coalesce(a.text, a.content_redacted->>'text', '') AS text,
            json_agg(t.label) FILTER (WHERE t.label IS NOT NULL) AS tags
       FROM anonymised_sample a
       LEFT JOIN sample_tag t ON t.sample_id = a.sample_id
      WHERE a.sample_id = $1
      GROUP BY a.sample_id`
    , [sampleId]
  );

  const row = rows[0];
  if (!row) return null;

  return {
    sample_id: row.sample_id,
    pseudo_student_id: row.pseudo_student_id,
    derived_at: row.derived_at,
    text: row.text,
    tags: row.tags ?? [],
  };
}

export async function listComments(sampleId: string) {
  const { rows } = await pool.query(
    `SELECT c.comment_id,
            c.text,
            c.created_at,
            u.name as author_name
       FROM sample_comment c
       JOIN app_user u ON c.author_id = u.user_id
      WHERE c.sample_id = $1
      ORDER BY c.created_at ASC`,
    [sampleId]
  );
  return rows;
}

export async function addTag(sampleId: string, label: string) {
  await pool.query(
    'INSERT INTO sample_tag (sample_id, label) VALUES ($1, $2)',
    [sampleId, label]
  );
}

export async function addComment(
  sampleId: string,
  authorId: string,
  text: string
) {
  const { rows } = await pool.query(
    `INSERT INTO sample_comment (sample_id, author_id, text)
     VALUES ($1, $2, $3)
     RETURNING comment_id, created_at`,
    [sampleId, authorId, text]
  );
  return rows[0];
}
