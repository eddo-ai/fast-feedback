/* ─────────────────────────────────────────────────────────
   ONE-SHOT SCHEMA + SEED
   (works on Supabase or vanilla Postgres)
   ───────────────────────────────────────────────────────── */

-- Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ENUMs
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'author_type_enum') THEN
    CREATE TYPE author_type_enum AS ENUM ('AI', 'Teacher', 'Peer');
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'artifact_status') THEN
    CREATE TYPE artifact_status AS ENUM ('draft', 'submitted', 'reviewed', 'archived');
  END IF;
END$$;

-- Templates
CREATE TABLE assessment_templates (
  template_id  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  title        TEXT NOT NULL,
  instructions TEXT,
  criteria     JSONB,
  metadata     JSONB,
  created_at   TIMESTAMPTZ DEFAULT now()
);

-- Artifacts
CREATE TABLE artifacts (
  artifact_id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  template_id          UUID REFERENCES assessment_templates(template_id) ON DELETE SET NULL,
  teacher_id           UUID,
  student_explanations JSONB NOT NULL,
  metadata             JSONB,
  status               artifact_status DEFAULT 'draft',
  created_at           TIMESTAMPTZ DEFAULT now()
);

-- Revision history
CREATE TABLE artifact_revisions (
  revision_id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  artifact_id          UUID REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
  version_number       INT NOT NULL,
  modified_by          UUID,
  modified_at          TIMESTAMPTZ DEFAULT now(),
  explanation_snapshot JSONB,
  metadata_snapshot    JSONB
);

-- Trigger to auto-log revisions
CREATE OR REPLACE FUNCTION log_artifact_revision()
RETURNS TRIGGER AS $$
DECLARE next_version INT;
BEGIN
  SELECT COALESCE(MAX(version_number),0)+1
  INTO   next_version
  FROM   artifact_revisions
  WHERE  artifact_id = NEW.artifact_id;

  INSERT INTO artifact_revisions (
    artifact_id, version_number, modified_by,
    explanation_snapshot, metadata_snapshot
  )
  VALUES (
    NEW.artifact_id, next_version, NEW.teacher_id,
    OLD.student_explanations, OLD.metadata
  );

  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_log_artifact_revision
AFTER UPDATE ON artifacts
FOR EACH ROW
WHEN (
  OLD.student_explanations IS DISTINCT FROM NEW.student_explanations OR
  OLD.metadata             IS DISTINCT FROM NEW.metadata             OR
  OLD.status               IS DISTINCT FROM NEW.status
)
EXECUTE FUNCTION log_artifact_revision();

-- Sense-making threads
CREATE TABLE sensemaking_threads (
  thread_id   UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  artifact_id UUID REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
  question_id TEXT,
  focus_tag   TEXT NOT NULL,
  created_by  UUID,
  created_at  TIMESTAMPTZ DEFAULT now()
);

-- Revisits (conversation entries)
CREATE TABLE revisits (
  revisit_id       UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  thread_id        UUID REFERENCES sensemaking_threads(thread_id) ON DELETE CASCADE,
  artifact_version INT,
  author_id        UUID NOT NULL,
  author_type      author_type_enum NOT NULL,
  content          TEXT NOT NULL,
  created_at       TIMESTAMPTZ DEFAULT now()
);

/* ─────────────────────
   SEED DATA
   ───────────────────── */

-- 1. Template
INSERT INTO assessment_templates (
  template_id, title, instructions, criteria, metadata
) VALUES (
  '00000000-0000-0000-0000-000000000001',
  'Lizard Behavior & Adaptation',
  'Explain how green lizards adapted to the presence of brown lizards.',
  '[{"tag":"Claim","description":"Makes a clear claim"}]'::jsonb,
  '{"unit":"6.1"}'::jsonb
);

-- 2. Artifact (v0)
INSERT INTO artifacts (
  artifact_id, template_id, teacher_id,
  student_explanations, metadata, status
) VALUES (
  '11111111-1111-1111-1111-111111111111',
  '00000000-0000-0000-0000-000000000001',
  '22222222-2222-2222-2222-222222222222',
  '[
     {
       "learner_id"  : "33333333-3333-3333-3333-333333333333",
       "question"    : "Q1",
       "explanation" : "Green lizards climb higher when brown lizards are present.",
       "feedback"    : []
     }
   ]'::jsonb,
  '{
     "learner_id"   : "33333333-3333-3333-3333-333333333333",
     "submitted_at" : "2025-05-29T12:00:00Z"
   }'::jsonb,
  'submitted'
);

-- 3. Manual revision snapshot (version 1)
INSERT INTO artifact_revisions (
  revision_id,
  artifact_id,
  version_number,
  modified_by,
  explanation_snapshot,
  metadata_snapshot
)
SELECT
  '55555555-5555-5555-5555-555555555555',
  artifact_id,
  1,
  teacher_id,
  student_explanations,
  metadata
FROM artifacts
WHERE artifact_id = '11111111-1111-1111-1111-111111111111';

-- 4. Sense-making thread
INSERT INTO sensemaking_threads (
  thread_id, artifact_id, question_id, focus_tag, created_by
) VALUES (
  '44444444-4444-4444-4444-444444444444',
  '11111111-1111-1111-1111-111111111111',
  'Q1',
  'reasoning',
  '22222222-2222-2222-2222-222222222222'  -- Teacher
);

-- 5. Two revisits
INSERT INTO revisits (
  revisit_id, thread_id, artifact_version,
  author_id, author_type, content
) VALUES
  ('66666666-6666-6666-6666-666666666666',
   '44444444-4444-4444-4444-444444444444',
   1,
   '22222222-2222-2222-2222-222222222222',
   'Teacher',
   'Let’s tighten the causal language connecting toe-pad size to survival advantage.'
  ),
  ('77777777-7777-7777-7777-777777777777',
   '44444444-4444-4444-4444-444444444444',
   1,
   '88888888-8888-8888-8888-888888888888',  -- AI agent UUID
   'AI',
   'Suggestion: add an explicit sentence that links selective pressure to genetic inheritance.'
  );