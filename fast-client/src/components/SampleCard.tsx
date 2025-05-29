"use client";

import Link from "next/link";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

export interface SampleCardProps {
  sample_id: string;
  pseudo_student_id: string;
  text: string;
  tags: string[];
}

export function SampleCard({ sample_id, pseudo_student_id, text, tags }: SampleCardProps) {
  return (
    <Link href={`/samples/${sample_id}`}>
      <Card className="hover:bg-muted/50 transition-colors">
        <CardContent className="space-y-2 p-4">
          <p className="text-gray-500 text-xs">Anon ID: {pseudo_student_id}</p>
          <p className="text-sm line-clamp-4">{text}</p>
          <div className="flex flex-wrap gap-1">
            {tags.map((tag) => (
              <Badge key={tag}>{tag}</Badge>
            ))}
          </div>
        </CardContent>
      </Card>
    </Link>
  );
}
