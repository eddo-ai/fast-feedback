import { NextResponse } from 'next/server';
import { addTag } from '@/lib/db';

export async function POST(
  req: Request,
  { params }: { params: { sample_id: string } }
) {
  const { label } = await req.json();
  await addTag(params.sample_id, label);
  return NextResponse.json({ ok: true });
}
