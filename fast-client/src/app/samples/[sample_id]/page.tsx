import TagManager from '@/components/TagManager';
import { CommentThread } from '@/components/CommentThread';
import { getSampleDetail, listComments } from '@/lib/db';

export default async function SampleDetail({
  params,
}: {
  params: { sample_id: string };
}) {
  const sample = await getSampleDetail(params.sample_id);
  if (!sample) {
    return <div>Sample not found</div>;
  }
  const comments = await listComments(params.sample_id);
  return (
    <div className="space-y-4">
      <p className="text-xs text-gray-500">
        Anon ID: {sample.pseudo_student_id} •
        {new Date(sample.derived_at).toLocaleString()}
      </p>
      <div className="h-48 overflow-y-auto rounded border p-4 text-sm whitespace-pre-wrap">
        {sample.text}
      </div>
      <TagManager sampleId={sample.sample_id} initialTags={sample.tags} />
      <CommentThread sampleId={sample.sample_id} initialComments={comments} />
    </div>
  );
}
