'use client';
import { useEffect, useState } from 'react';
import { Button } from '@/components/ui/button';

export interface Comment {
  comment_id: string;
  text: string;
  created_at: string;
  author_name: string;
}

interface ThreadProps {
  sampleId: string;
  initialComments: Comment[];
  editable?: boolean;
}

export function CommentThread({ sampleId, initialComments, editable = true }: ThreadProps) {
  const [comments, setComments] = useState<Comment[]>(initialComments);
  const [value, setValue] = useState('');

  // Placeholder for realtime subscription
  useEffect(() => {
    setComments(initialComments);
  }, [initialComments]);

  async function submit() {
    const text = value.trim();
    if (!text) return;
    setValue('');
    const res = await fetch(`/api/samples/${sampleId}/comments`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    const comment = await res.json();
    setComments([...comments, comment]);
  }

  return (
    <div className="space-y-3">
      <div className="space-y-2">
        {comments.length ? (
          comments.map((c) => (
            <div key={c.comment_id} className="rounded border p-2">
              <p className="text-sm">{c.text}</p>
              <p className="text-xs text-muted-foreground">
                {c.author_name} • {new Date(c.created_at).toLocaleString()}
              </p>
            </div>
          ))
        ) : (
          <p className="text-sm text-muted-foreground">Be the first to comment</p>
        )}
      </div>
      {editable && (
        <div className="space-y-2">
          <textarea
            className="w-full rounded border p-2 text-sm"
            rows={3}
            value={value}
            onChange={(e) => setValue(e.target.value)}
          />
          <Button type="button" onClick={submit} size="sm">
            Submit
          </Button>
        </div>
      )}
    </div>
  );
}
