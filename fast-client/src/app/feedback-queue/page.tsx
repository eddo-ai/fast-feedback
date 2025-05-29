import FeedbackQueueTable from "@/components/FeedbackQueueTable";
import { getFeedbackQueue } from "@/lib/db";

export default async function FeedbackQueuePage() {
  // Placeholder user id until auth added
  const drafts = await getFeedbackQueue("00000000-0000-0000-0000-000000000000");
  return (
    <div className="space-y-4">
      <h1 className="text-xl font-bold">Feedback Queue</h1>
      <FeedbackQueueTable drafts={drafts} />
    </div>
  );
}
