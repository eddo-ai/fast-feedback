'use client';
import { useState } from 'react';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Tag } from 'lucide-react';

interface Props {
  sampleId: string;
  initialTags: string[];
  editable?: boolean;
}

export default function TagManager({ sampleId, initialTags, editable = true }: Props) {
  const [tags, setTags] = useState<string[]>(initialTags);
  const [value, setValue] = useState('');

  async function addTag() {
    const label = value.trim();
    if (!label) return;
    setTags([...tags, label]);
    setValue('');
    try {
      await fetch(`/api/samples/${sampleId}/tags`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ label }),
      });
    } catch {
      // ignore network errors for demo
    }
  }

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-1">
        {tags.length ? (
          tags.map((t) => <Badge key={t}>{t}</Badge>)
        ) : (
          <p className="text-sm text-muted-foreground">No tags yet</p>
        )}
      </div>
      {editable && (
        <div className="flex gap-2">
          <Input
            value={value}
            onChange={(e) => setValue(e.target.value)}
            placeholder="Add tag"
            className="flex-1"
          />
          <Button type="button" onClick={addTag} size="sm">
            <Tag className="w-4 h-4 mr-1" /> Add
          </Button>
        </div>
      )}
    </div>
  );
}
