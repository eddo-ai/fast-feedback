import { supa } from "@/lib/supa";
import ArtifactTable from "@/components/artifact-table";

export default async function ArtifactsPage() {
  const { data } = await supa
    .from("artifacts")
    .select(
      "artifact_id, metadata->>learner_id, metadata->>submitted_at, status",
    )
    .order("created_at", { ascending: false });

  return (
    <div className="p-6">
      <h1 className="text-2xl font-semibold mb-4">Artifacts</h1>
      <ArtifactTable rows={data ?? []} />
    </div>
  );
}
