import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

export default function Home() {
  return (
    <Card className="mx-auto max-w-md">
      <CardContent className="space-y-4 p-6">
        <h1 className="text-2xl font-bold">Eddo Learning</h1>
        <p className="text-sm text-gray-600">
          Connected to your new Postgres schema.
        </p>
        <Button asChild>
          <a href="/samples">View Sample Gallery</a>
        </Button>
      </CardContent>
    </Card>
  );
}
