import { NextResponse } from 'next/server';
import { addComment } from '@/lib/db';

export async function POST(
  req: Request,
  { params }: { params: { sample_id: string } }
) {
  const { text, authorId } = await req.json();
  const result = await addComment(params.sample_id, authorId || '00000000-0000-0000-0000-000000000000', text);
  return NextResponse.json({
    comment_id: result.comment_id,
    text,
    created_at: result.created_at,
    author_name: 'You',
  });
}
