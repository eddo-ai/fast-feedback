"use client";

import { useEffect, useState } from "react";
import { SampleCard } from "@/components/SampleCard";
import { listSamples } from "@/lib/db";
import { Card, CardContent } from "@/components/ui/card";

interface Sample {
  sample_id: string;
  pseudo_student_id: string;
  text: string;
  tags: string[];
}

export default function Samples() {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    listSamples()
      .then(setSamples)
      .catch(() => setError(true))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="grid gap-4 md:grid-cols-2">
        {Array.from({ length: 2 }).map((_, i) => (
          <div key={i} className="h-32 rounded-lg bg-gray-200 animate-pulse" />
        ))}
      </div>
    );
  }

  if (error) {
    return <p className="text-center text-red-500">Failed to load samples.</p>;
  }

  if (samples.length === 0) {
    return (
      <Card className="mx-auto max-w-md">
        <CardContent className="p-6 text-center">
          No samples yet—share one from Submissions.
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="grid gap-4 md:grid-cols-2">
      {samples.map((sample) => (
        <SampleCard key={sample.sample_id} {...sample} />
      ))}
    </div>
  );
}
