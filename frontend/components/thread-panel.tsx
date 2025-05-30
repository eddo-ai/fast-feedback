"use client";

import { useEffect, useState } from "react";
import { supa } from "@/lib/supa";
import type { Thread, Revisit } from "@/lib/types";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import RevisitList from "@/components/revisit-list";

export default function ThreadPanel({ thread }: { thread: Thread }) {
  const [revisits, setRevisits] = useState<Revisit[]>([]);
  const [text, setText] = useState("");

  useEffect(() => {
    supa
      .from("revisits")
      .select("*")
      .eq("thread_id", thread.thread_id)
      .order("created_at")
      .then(({ data }) => setRevisits(data ?? []));
  }, [thread.thread_id]);

  async function addEntry() {
    if (!text) return;
    const { data, error } = await supa.from("revisits").insert({
      thread_id: thread.thread_id,
      artifact_version: 1,
      author_id: "demo-user-id", // replace with auth.user.id
      author_type: "Peer",
      content: text,
    });
    if (!error) setRevisits((prev) => [...prev, ...(data ?? [])]);
    setText("");
  }

  return (
    <div className="border rounded p-4 space-y-4">
      <h2 className="font-semibold">Sense-making thread: {thread.focus_tag}</h2>
      <RevisitList revisits={revisits} />
      <Textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Add your thinking…"
      />
      <Button onClick={addEntry}>Post</Button>
    </div>
  );
}
