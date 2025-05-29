import { NextResponse } from "next/server";
import { Pool } from "pg";
import { supabase } from '@/lib/supabase';

const pool = new Pool({
  connectionString: process.env.POSTGRES_URL,
});

export async function GET() {
  // Using Supabase client
  const { data, error } = await supabase
    .from('anonymised_sample')
    .select(`
      sample_id,
      pseudo_student_id,
      content_redacted->>text,
      tag_link:tag_link!left(target_id,sample_id,target_type),
      skill_tag:skill_tag!left(tag_id,tag_link.tag_id,name)
    `);

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  // Optionally, transform data to match the original shape
  // ...

  // Fallback to original pool query if needed
  // const { rows } = await pool.query(...)
  // return NextResponse.json(rows);

  return NextResponse.json(data);
}
