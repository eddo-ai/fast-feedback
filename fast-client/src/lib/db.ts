import { supabase } from "./supabase";

export interface Sample {
  sample_id: string;
  pseudo_student_id: string;
  text: string;
  tags: string[];
}

export async function listSamples(): Promise<Sample[]> {
  const { data, error } = await supabase
    .from("anonymised_sample")
    .select("sample_id,pseudo_student_id,content_redacted,sample_tag(label)");

  if (error) throw error;

  return (
    data?.map((row) => ({
      sample_id: row.sample_id as string,
      pseudo_student_id: row.pseudo_student_id as string,
      text: ((row.content_redacted as {text?: string}|null)?.text ?? ""),
      tags: ((row.sample_tag as {label: string}[]|null)?.map(t => t.label) ?? []),
    })) ?? []
  );
}
