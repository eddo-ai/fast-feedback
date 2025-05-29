"use client";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { publishFeedback } from "@/lib/db";

type Draft = {
  feedback_id: string;
  submission_id: string;
  student_name: string;
  assignment_title: string;
  draft: string;
};

type Props = { drafts: Draft[] };

export default function FeedbackQueueTable({ drafts }: Props) {
  const [items, setItems] = useState<Draft[]>(drafts);

  async function quickApprove(id: string) {
    const remaining = items.filter((d) => d.feedback_id !== id);
    setItems(remaining);
    try {
      await publishFeedback(id);
    } catch (err) {
      console.error(err);
      setItems(items);
    }
  }

  if (items.length === 0) {
    return (
      <div className="text-center p-8 border rounded">
        No feedback waiting — 🎉
      </div>
    );
  }

  return (
    <table className="w-full border-collapse">
      <thead>
        <tr>
          <th className="text-left p-2 border">Student</th>
          <th className="text-left p-2 border">Assignment</th>
          <th className="text-left p-2 border">Draft</th>
          <th className="p-2 border">Action</th>
        </tr>
      </thead>
      <tbody>
        {items.map((d) => (
          <tr key={d.feedback_id} className="align-top">
            <td className="p-2 border">{d.student_name || "Anon"}</td>
            <td className="p-2 border">{d.assignment_title}</td>
            <td className="p-2 border max-w-sm">
              {d.draft.slice(0, 120)}
              {d.draft.length > 120 ? "…" : ""}
            </td>
            <td className="p-2 border text-right">
              <Button onClick={() => quickApprove(d.feedback_id)} size="sm">
                Quick Approve
              </Button>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
