import { NextResponse } from "next/server";
import { Pool } from "pg";

const pool = new Pool({
  connectionString: process.env.POSTGRES_URL,
});

export async function GET() {
  const { rows } = await pool.query('SELECT * FROM anonymised_sample');
  return NextResponse.json(rows);
}
