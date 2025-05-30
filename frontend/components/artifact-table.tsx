"use client";

import Link from "next/link";
import type { Artifact } from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

export default function ArtifactTable({ rows }: { rows: Artifact[] }) {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>ID</TableHead>
          <TableHead>Learner</TableHead>
          <TableHead>Submitted</TableHead>
          <TableHead>Status</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {rows.map((r) => (
          <TableRow key={r.artifact_id} className="cursor-pointer hover:bg-muted/50">
            <TableCell>
              <Link href={`/artifacts/${r.artifact_id}`} className="underline">
                {r.artifact_id.slice(0, 8)}…
              </Link>
            </TableCell>
            <TableCell>{r.learner_id.slice(0, 8)}</TableCell>
            <TableCell>
              {new Date(r.submitted_at).toLocaleDateString(undefined, {
                month: "short",
                day: "numeric",
                year: "numeric",
              })}
            </TableCell>
            <TableCell>
              <Badge variant="outline">{r.status}</Badge>
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
