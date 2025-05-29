import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { listSamples } from "@/lib/db";

export default async function Samples() {
  const samples = await listSamples();

  return (
    <div className="grid gap-4 md:grid-cols-2">
      {samples.map(({ sample_id, pseudo_student_id, text, tags }) => (
        <Card key={sample_id}>
          <CardContent className="space-y-2 p-4">
            <p className="text-gray-500 text-xs">Anon ID: {pseudo_student_id}</p>
            <p className="text-sm line-clamp-4">{text}</p>
            <div className="flex flex-wrap gap-1">
              {tags.map((tag: string) => (
                <Badge key={tag}>{tag}</Badge>
              ))}
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
