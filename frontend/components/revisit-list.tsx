"use client";

import type { Revisit } from "@/lib/types";

export default function RevisitList({ revisits }: { revisits: Revisit[] }) {
  return (
    <div className="space-y-2 max-h-72 overflow-y-auto">
      {revisits.map((r) => (
        <div key={r.revisit_id} className="border p-2 rounded">
          <span className="text-xs text-muted-foreground">{r.author_type}</span>
          <p>{r.content}</p>
        </div>
      ))}
    </div>
  );
}
