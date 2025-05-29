export async function listSamples() {
  const res = await fetch("/api/samples");
  if (!res.ok) throw new Error("Failed to fetch");
  return (await res.json()) as {
    sample_id: string;
    pseudo_student_id: string;
    text: string;
    tags: string[];
  }[];
}
