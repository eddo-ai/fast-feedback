create type "public"."feedback_status" as enum ('needs_review', 'approved', 'rejected');

create type "public"."feedback_type" as enum ('teacher_written', 'ai_suggested', 'student_revision');

create table "public"."anonymised_sample" (
    "sample_id" uuid not null default gen_random_uuid(),
    "set_id" uuid,
    "source_submission_id" uuid,
    "pseudo_student_id" text,
    "content_redacted" jsonb,
    "derived_at" timestamp with time zone default now()
);


create table "public"."app_user" (
    "user_id" uuid not null default gen_random_uuid(),
    "name" text not null,
    "role" text not null
);


create table "public"."assignment" (
    "assignment_id" uuid not null default gen_random_uuid(),
    "section_id" uuid,
    "title" text not null
);


create table "public"."class_section" (
    "section_id" uuid not null default gen_random_uuid(),
    "name" text not null
);


create table "public"."narrative_feedback" (
    "feedback_id" uuid not null default gen_random_uuid(),
    "submission_id" uuid,
    "author_id" uuid,
    "feedback_type" feedback_type,
    "feedback_status" feedback_status,
    "text" text,
    "created_at" timestamp with time zone default now()
);


create table "public"."sample_set" (
    "set_id" uuid not null default gen_random_uuid(),
    "team_id" uuid,
    "title" text
);


create table "public"."sample_tag" (
    "tag_id" uuid not null default gen_random_uuid(),
    "sample_id" uuid,
    "label" text not null
);


create table "public"."student_submission" (
    "submission_id" uuid not null default gen_random_uuid(),
    "assignment_id" uuid,
    "student_id" uuid,
    "text" text not null,
    "submitted_at" timestamp with time zone default now()
);


create table "public"."team" (
    "team_id" uuid not null default gen_random_uuid(),
    "name" text not null
);


create table "public"."team_member" (
    "user_id" uuid not null,
    "team_id" uuid not null,
    "role" text
);


CREATE UNIQUE INDEX anonymised_sample_pkey ON public.anonymised_sample USING btree (sample_id);

CREATE UNIQUE INDEX app_user_pkey ON public.app_user USING btree (user_id);

CREATE UNIQUE INDEX assignment_pkey ON public.assignment USING btree (assignment_id);

CREATE UNIQUE INDEX class_section_pkey ON public.class_section USING btree (section_id);

CREATE UNIQUE INDEX narrative_feedback_pkey ON public.narrative_feedback USING btree (feedback_id);

CREATE UNIQUE INDEX sample_set_pkey ON public.sample_set USING btree (set_id);

CREATE UNIQUE INDEX sample_tag_pkey ON public.sample_tag USING btree (tag_id);

CREATE UNIQUE INDEX student_submission_pkey ON public.student_submission USING btree (submission_id);

CREATE UNIQUE INDEX team_member_pkey ON public.team_member USING btree (user_id, team_id);

CREATE UNIQUE INDEX team_pkey ON public.team USING btree (team_id);

alter table "public"."anonymised_sample" add constraint "anonymised_sample_pkey" PRIMARY KEY using index "anonymised_sample_pkey";

alter table "public"."app_user" add constraint "app_user_pkey" PRIMARY KEY using index "app_user_pkey";

alter table "public"."assignment" add constraint "assignment_pkey" PRIMARY KEY using index "assignment_pkey";

alter table "public"."class_section" add constraint "class_section_pkey" PRIMARY KEY using index "class_section_pkey";

alter table "public"."narrative_feedback" add constraint "narrative_feedback_pkey" PRIMARY KEY using index "narrative_feedback_pkey";

alter table "public"."sample_set" add constraint "sample_set_pkey" PRIMARY KEY using index "sample_set_pkey";

alter table "public"."sample_tag" add constraint "sample_tag_pkey" PRIMARY KEY using index "sample_tag_pkey";

alter table "public"."student_submission" add constraint "student_submission_pkey" PRIMARY KEY using index "student_submission_pkey";

alter table "public"."team" add constraint "team_pkey" PRIMARY KEY using index "team_pkey";

alter table "public"."team_member" add constraint "team_member_pkey" PRIMARY KEY using index "team_member_pkey";

alter table "public"."anonymised_sample" add constraint "anonymised_sample_set_id_fkey" FOREIGN KEY (set_id) REFERENCES sample_set(set_id) not valid;

alter table "public"."anonymised_sample" validate constraint "anonymised_sample_set_id_fkey";

alter table "public"."anonymised_sample" add constraint "anonymised_sample_source_submission_id_fkey" FOREIGN KEY (source_submission_id) REFERENCES student_submission(submission_id) not valid;

alter table "public"."anonymised_sample" validate constraint "anonymised_sample_source_submission_id_fkey";

alter table "public"."assignment" add constraint "assignment_section_id_fkey" FOREIGN KEY (section_id) REFERENCES class_section(section_id) not valid;

alter table "public"."assignment" validate constraint "assignment_section_id_fkey";

alter table "public"."narrative_feedback" add constraint "narrative_feedback_author_id_fkey" FOREIGN KEY (author_id) REFERENCES app_user(user_id) not valid;

alter table "public"."narrative_feedback" validate constraint "narrative_feedback_author_id_fkey";

alter table "public"."narrative_feedback" add constraint "narrative_feedback_submission_id_fkey" FOREIGN KEY (submission_id) REFERENCES student_submission(submission_id) not valid;

alter table "public"."narrative_feedback" validate constraint "narrative_feedback_submission_id_fkey";

alter table "public"."sample_set" add constraint "sample_set_team_id_fkey" FOREIGN KEY (team_id) REFERENCES team(team_id) not valid;

alter table "public"."sample_set" validate constraint "sample_set_team_id_fkey";

alter table "public"."sample_tag" add constraint "sample_tag_sample_id_fkey" FOREIGN KEY (sample_id) REFERENCES anonymised_sample(sample_id) not valid;

alter table "public"."sample_tag" validate constraint "sample_tag_sample_id_fkey";

alter table "public"."student_submission" add constraint "student_submission_assignment_id_fkey" FOREIGN KEY (assignment_id) REFERENCES assignment(assignment_id) not valid;

alter table "public"."student_submission" validate constraint "student_submission_assignment_id_fkey";

alter table "public"."student_submission" add constraint "student_submission_student_id_fkey" FOREIGN KEY (student_id) REFERENCES app_user(user_id) not valid;

alter table "public"."student_submission" validate constraint "student_submission_student_id_fkey";

alter table "public"."team_member" add constraint "team_member_team_id_fkey" FOREIGN KEY (team_id) REFERENCES team(team_id) not valid;

alter table "public"."team_member" validate constraint "team_member_team_id_fkey";

alter table "public"."team_member" add constraint "team_member_user_id_fkey" FOREIGN KEY (user_id) REFERENCES app_user(user_id) not valid;

alter table "public"."team_member" validate constraint "team_member_user_id_fkey";

grant delete on table "public"."anonymised_sample" to "anon";

grant insert on table "public"."anonymised_sample" to "anon";

grant references on table "public"."anonymised_sample" to "anon";

grant select on table "public"."anonymised_sample" to "anon";

grant trigger on table "public"."anonymised_sample" to "anon";

grant truncate on table "public"."anonymised_sample" to "anon";

grant update on table "public"."anonymised_sample" to "anon";

grant delete on table "public"."anonymised_sample" to "authenticated";

grant insert on table "public"."anonymised_sample" to "authenticated";

grant references on table "public"."anonymised_sample" to "authenticated";

grant select on table "public"."anonymised_sample" to "authenticated";

grant trigger on table "public"."anonymised_sample" to "authenticated";

grant truncate on table "public"."anonymised_sample" to "authenticated";

grant update on table "public"."anonymised_sample" to "authenticated";

grant delete on table "public"."anonymised_sample" to "service_role";

grant insert on table "public"."anonymised_sample" to "service_role";

grant references on table "public"."anonymised_sample" to "service_role";

grant select on table "public"."anonymised_sample" to "service_role";

grant trigger on table "public"."anonymised_sample" to "service_role";

grant truncate on table "public"."anonymised_sample" to "service_role";

grant update on table "public"."anonymised_sample" to "service_role";

grant delete on table "public"."app_user" to "anon";

grant insert on table "public"."app_user" to "anon";

grant references on table "public"."app_user" to "anon";

grant select on table "public"."app_user" to "anon";

grant trigger on table "public"."app_user" to "anon";

grant truncate on table "public"."app_user" to "anon";

grant update on table "public"."app_user" to "anon";

grant delete on table "public"."app_user" to "authenticated";

grant insert on table "public"."app_user" to "authenticated";

grant references on table "public"."app_user" to "authenticated";

grant select on table "public"."app_user" to "authenticated";

grant trigger on table "public"."app_user" to "authenticated";

grant truncate on table "public"."app_user" to "authenticated";

grant update on table "public"."app_user" to "authenticated";

grant delete on table "public"."app_user" to "service_role";

grant insert on table "public"."app_user" to "service_role";

grant references on table "public"."app_user" to "service_role";

grant select on table "public"."app_user" to "service_role";

grant trigger on table "public"."app_user" to "service_role";

grant truncate on table "public"."app_user" to "service_role";

grant update on table "public"."app_user" to "service_role";

grant delete on table "public"."assignment" to "anon";

grant insert on table "public"."assignment" to "anon";

grant references on table "public"."assignment" to "anon";

grant select on table "public"."assignment" to "anon";

grant trigger on table "public"."assignment" to "anon";

grant truncate on table "public"."assignment" to "anon";

grant update on table "public"."assignment" to "anon";

grant delete on table "public"."assignment" to "authenticated";

grant insert on table "public"."assignment" to "authenticated";

grant references on table "public"."assignment" to "authenticated";

grant select on table "public"."assignment" to "authenticated";

grant trigger on table "public"."assignment" to "authenticated";

grant truncate on table "public"."assignment" to "authenticated";

grant update on table "public"."assignment" to "authenticated";

grant delete on table "public"."assignment" to "service_role";

grant insert on table "public"."assignment" to "service_role";

grant references on table "public"."assignment" to "service_role";

grant select on table "public"."assignment" to "service_role";

grant trigger on table "public"."assignment" to "service_role";

grant truncate on table "public"."assignment" to "service_role";

grant update on table "public"."assignment" to "service_role";

grant delete on table "public"."class_section" to "anon";

grant insert on table "public"."class_section" to "anon";

grant references on table "public"."class_section" to "anon";

grant select on table "public"."class_section" to "anon";

grant trigger on table "public"."class_section" to "anon";

grant truncate on table "public"."class_section" to "anon";

grant update on table "public"."class_section" to "anon";

grant delete on table "public"."class_section" to "authenticated";

grant insert on table "public"."class_section" to "authenticated";

grant references on table "public"."class_section" to "authenticated";

grant select on table "public"."class_section" to "authenticated";

grant trigger on table "public"."class_section" to "authenticated";

grant truncate on table "public"."class_section" to "authenticated";

grant update on table "public"."class_section" to "authenticated";

grant delete on table "public"."class_section" to "service_role";

grant insert on table "public"."class_section" to "service_role";

grant references on table "public"."class_section" to "service_role";

grant select on table "public"."class_section" to "service_role";

grant trigger on table "public"."class_section" to "service_role";

grant truncate on table "public"."class_section" to "service_role";

grant update on table "public"."class_section" to "service_role";

grant delete on table "public"."narrative_feedback" to "anon";

grant insert on table "public"."narrative_feedback" to "anon";

grant references on table "public"."narrative_feedback" to "anon";

grant select on table "public"."narrative_feedback" to "anon";

grant trigger on table "public"."narrative_feedback" to "anon";

grant truncate on table "public"."narrative_feedback" to "anon";

grant update on table "public"."narrative_feedback" to "anon";

grant delete on table "public"."narrative_feedback" to "authenticated";

grant insert on table "public"."narrative_feedback" to "authenticated";

grant references on table "public"."narrative_feedback" to "authenticated";

grant select on table "public"."narrative_feedback" to "authenticated";

grant trigger on table "public"."narrative_feedback" to "authenticated";

grant truncate on table "public"."narrative_feedback" to "authenticated";

grant update on table "public"."narrative_feedback" to "authenticated";

grant delete on table "public"."narrative_feedback" to "service_role";

grant insert on table "public"."narrative_feedback" to "service_role";

grant references on table "public"."narrative_feedback" to "service_role";

grant select on table "public"."narrative_feedback" to "service_role";

grant trigger on table "public"."narrative_feedback" to "service_role";

grant truncate on table "public"."narrative_feedback" to "service_role";

grant update on table "public"."narrative_feedback" to "service_role";

grant delete on table "public"."sample_set" to "anon";

grant insert on table "public"."sample_set" to "anon";

grant references on table "public"."sample_set" to "anon";

grant select on table "public"."sample_set" to "anon";

grant trigger on table "public"."sample_set" to "anon";

grant truncate on table "public"."sample_set" to "anon";

grant update on table "public"."sample_set" to "anon";

grant delete on table "public"."sample_set" to "authenticated";

grant insert on table "public"."sample_set" to "authenticated";

grant references on table "public"."sample_set" to "authenticated";

grant select on table "public"."sample_set" to "authenticated";

grant trigger on table "public"."sample_set" to "authenticated";

grant truncate on table "public"."sample_set" to "authenticated";

grant update on table "public"."sample_set" to "authenticated";

grant delete on table "public"."sample_set" to "service_role";

grant insert on table "public"."sample_set" to "service_role";

grant references on table "public"."sample_set" to "service_role";

grant select on table "public"."sample_set" to "service_role";

grant trigger on table "public"."sample_set" to "service_role";

grant truncate on table "public"."sample_set" to "service_role";

grant update on table "public"."sample_set" to "service_role";

grant delete on table "public"."sample_tag" to "anon";

grant insert on table "public"."sample_tag" to "anon";

grant references on table "public"."sample_tag" to "anon";

grant select on table "public"."sample_tag" to "anon";

grant trigger on table "public"."sample_tag" to "anon";

grant truncate on table "public"."sample_tag" to "anon";

grant update on table "public"."sample_tag" to "anon";

grant delete on table "public"."sample_tag" to "authenticated";

grant insert on table "public"."sample_tag" to "authenticated";

grant references on table "public"."sample_tag" to "authenticated";

grant select on table "public"."sample_tag" to "authenticated";

grant trigger on table "public"."sample_tag" to "authenticated";

grant truncate on table "public"."sample_tag" to "authenticated";

grant update on table "public"."sample_tag" to "authenticated";

grant delete on table "public"."sample_tag" to "service_role";

grant insert on table "public"."sample_tag" to "service_role";

grant references on table "public"."sample_tag" to "service_role";

grant select on table "public"."sample_tag" to "service_role";

grant trigger on table "public"."sample_tag" to "service_role";

grant truncate on table "public"."sample_tag" to "service_role";

grant update on table "public"."sample_tag" to "service_role";

grant delete on table "public"."student_submission" to "anon";

grant insert on table "public"."student_submission" to "anon";

grant references on table "public"."student_submission" to "anon";

grant select on table "public"."student_submission" to "anon";

grant trigger on table "public"."student_submission" to "anon";

grant truncate on table "public"."student_submission" to "anon";

grant update on table "public"."student_submission" to "anon";

grant delete on table "public"."student_submission" to "authenticated";

grant insert on table "public"."student_submission" to "authenticated";

grant references on table "public"."student_submission" to "authenticated";

grant select on table "public"."student_submission" to "authenticated";

grant trigger on table "public"."student_submission" to "authenticated";

grant truncate on table "public"."student_submission" to "authenticated";

grant update on table "public"."student_submission" to "authenticated";

grant delete on table "public"."student_submission" to "service_role";

grant insert on table "public"."student_submission" to "service_role";

grant references on table "public"."student_submission" to "service_role";

grant select on table "public"."student_submission" to "service_role";

grant trigger on table "public"."student_submission" to "service_role";

grant truncate on table "public"."student_submission" to "service_role";

grant update on table "public"."student_submission" to "service_role";

grant delete on table "public"."team" to "anon";

grant insert on table "public"."team" to "anon";

grant references on table "public"."team" to "anon";

grant select on table "public"."team" to "anon";

grant trigger on table "public"."team" to "anon";

grant truncate on table "public"."team" to "anon";

grant update on table "public"."team" to "anon";

grant delete on table "public"."team" to "authenticated";

grant insert on table "public"."team" to "authenticated";

grant references on table "public"."team" to "authenticated";

grant select on table "public"."team" to "authenticated";

grant trigger on table "public"."team" to "authenticated";

grant truncate on table "public"."team" to "authenticated";

grant update on table "public"."team" to "authenticated";

grant delete on table "public"."team" to "service_role";

grant insert on table "public"."team" to "service_role";

grant references on table "public"."team" to "service_role";

grant select on table "public"."team" to "service_role";

grant trigger on table "public"."team" to "service_role";

grant truncate on table "public"."team" to "service_role";

grant update on table "public"."team" to "service_role";

grant delete on table "public"."team_member" to "anon";

grant insert on table "public"."team_member" to "anon";

grant references on table "public"."team_member" to "anon";

grant select on table "public"."team_member" to "anon";

grant trigger on table "public"."team_member" to "anon";

grant truncate on table "public"."team_member" to "anon";

grant update on table "public"."team_member" to "anon";

grant delete on table "public"."team_member" to "authenticated";

grant insert on table "public"."team_member" to "authenticated";

grant references on table "public"."team_member" to "authenticated";

grant select on table "public"."team_member" to "authenticated";

grant trigger on table "public"."team_member" to "authenticated";

grant truncate on table "public"."team_member" to "authenticated";

grant update on table "public"."team_member" to "authenticated";

grant delete on table "public"."team_member" to "service_role";

grant insert on table "public"."team_member" to "service_role";

grant references on table "public"."team_member" to "service_role";

grant select on table "public"."team_member" to "service_role";

grant trigger on table "public"."team_member" to "service_role";

grant truncate on table "public"."team_member" to "service_role";

grant update on table "public"."team_member" to "service_role";


