
-- Schema and Seed Script for Eddo Learning Sample Setup

-- ENUMS
CREATE TYPE feedback_status AS ENUM ('needs_review', 'approved', 'rejected');
CREATE TYPE feedback_type AS ENUM ('teacher_written', 'ai_suggested', 'student_revision');

-- USERS
CREATE TABLE app_user (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    role TEXT NOT NULL
);

-- CLASSES
CREATE TABLE class_section (
    section_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL
);

-- ASSIGNMENTS
CREATE TABLE assignment (
    assignment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    section_id UUID REFERENCES class_section(section_id),
    title TEXT NOT NULL
);

-- STUDENT SUBMISSIONS
CREATE TABLE student_submission (
    submission_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    assignment_id UUID REFERENCES assignment(assignment_id),
    student_id UUID REFERENCES app_user(user_id),
    text TEXT NOT NULL,
    submitted_at TIMESTAMPTZ DEFAULT now()
);

-- FEEDBACK
CREATE TABLE narrative_feedback (
    feedback_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    submission_id UUID REFERENCES student_submission(submission_id),
    author_id UUID REFERENCES app_user(user_id),
    feedback_type feedback_type,
    feedback_status feedback_status,
    text TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- TEAMS
CREATE TABLE team (
    team_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL
);

CREATE TABLE team_member (
    user_id UUID REFERENCES app_user(user_id),
    team_id UUID REFERENCES team(team_id),
    role TEXT,
    PRIMARY KEY (user_id, team_id)
);

-- SAMPLE SETS
CREATE TABLE sample_set (
    set_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    team_id UUID REFERENCES team(team_id),
    title TEXT
);

-- ANONYMISED SAMPLES
CREATE TABLE anonymised_sample (
    sample_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    set_id UUID REFERENCES sample_set(set_id),
    source_submission_id UUID REFERENCES student_submission(submission_id),
    pseudo_student_id TEXT,
    content_redacted JSONB,
    derived_at TIMESTAMPTZ DEFAULT now()
);

-- TAGGING
CREATE TABLE sample_tag (
    tag_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sample_id UUID REFERENCES anonymised_sample(sample_id),
    label TEXT NOT NULL
);

-- SEED USERS
INSERT INTO app_user (name, role) VALUES
('Alice Teacher', 'teacher'),
('Bob Student', 'student'),
('Carol Coach', 'coach');

-- SEED CLASS
INSERT INTO class_section (name) VALUES ('Period 1 Science');

-- SEED ASSIGNMENT
INSERT INTO assignment (section_id, title)
SELECT section_id, 'Thermal Reactions' FROM class_section;

-- SEED SUBMISSION
INSERT INTO student_submission (assignment_id, student_id, text)
SELECT assignment.assignment_id, u.user_id, 'When metal gets hot, it glows and transfers energy.'
FROM assignment
JOIN app_user u ON u.name = 'Bob Student';

-- SEED FEEDBACK
INSERT INTO narrative_feedback (submission_id, author_id, feedback_type, feedback_status, text)
SELECT s.submission_id, u.user_id, 'ai_suggested', 'needs_review', 'Good start! Can you say more about how energy moves?'
FROM student_submission s, app_user u
WHERE u.name = 'Alice Teacher';

-- TEAM + MEMBERS
INSERT INTO team (name) VALUES ('6th Grade Science Team');
INSERT INTO team_member (user_id, team_id, role)
SELECT u.user_id, t.team_id, 'member'
FROM app_user u, team t
WHERE t.name = '6th Grade Science Team' AND u.name IN ('Alice Teacher', 'Carol Coach');

-- SAMPLE SET AND ANONYMISED SAMPLE
INSERT INTO sample_set (team_id, title)
SELECT team_id, 'Exemplars: Thermal Reactions' FROM team;

INSERT INTO anonymised_sample (set_id, source_submission_id, pseudo_student_id, content_redacted)
SELECT ss.set_id, s.submission_id, 'student-001', jsonb_build_object('text', s.text)
FROM sample_set ss, student_submission s;

-- TAGGING
INSERT INTO sample_tag (sample_id, label)
SELECT sample_id, 'Exemplar' FROM anonymised_sample;
