export type Artifact = {
  artifact_id: string;
  learner_id: string;
  submitted_at: string;
  status: "draft" | "submitted" | "reviewed" | "archived";
};

export type Thread = {
  thread_id: string;
  focus_tag: string;
  question_id: string | null;
};

export type Revisit = {
  revisit_id: string;
  author_type: "Teacher" | "AI" | "Peer";
  content: string;
  created_at: string;
};
