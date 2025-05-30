import { supa } from "@/lib/supa";
import ThreadPanel from "@/components/thread-panel";
import { notFound } from "next/navigation";

export default async function ArtifactDetail({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;

  const { data: artifact } = await supa
    .from("artifacts")
    .select("artifact_id, student_explanations, status")
    .eq("artifact_id", id)
    .single();

  if (!artifact) return notFound();

  const { data: thread } = await supa
    .from("sensemaking_threads")
    .select("thread_id, focus_tag, question_id")
    .eq("artifact_id", id)
    .limit(1)
    .maybeSingle();

  return (
    <div className="p-6 space-y-6">
      <pre className="bg-muted p-4 rounded">
        {JSON.stringify(artifact.student_explanations, null, 2)}
      </pre>
      {thread ? <ThreadPanel thread={thread} /> : <p>No thread yet.</p>}
    </div>
  );
}
