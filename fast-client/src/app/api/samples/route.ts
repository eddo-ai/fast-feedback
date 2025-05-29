import { NextResponse } from "next/server";
import { Pool } from "pg";

const pool = new Pool({
  connectionString: process.env.POSTGRES_URL,
});

export async function GET() {
  const { rows } = await pool.query(
    `
    SELECT s.sample_id,
           s.pseudo_student_id,
           s.content_redacted->>'text' AS text,
           array_agg(t.name) AS tags
    FROM anonymised_sample s
    LEFT JOIN tag_link tl ON tl.target_id = s.sample_id AND tl.target_type = 'anonymised_sample'
    LEFT JOIN skill_tag t ON t.tag_id = tl.tag_id
    GROUP BY s.sample_id;
  `
  );
  return NextResponse.json(rows);
}
