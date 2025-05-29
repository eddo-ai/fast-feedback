-- Example seed data for anonymised_sample and sample_tag
insert into public.anonymised_sample (pseudo_student_id, content_redacted)
values
  ('S001', '{"text": "This is an exemplar sample"}'),
  ('S002', '{"text": "This sample illustrates a misconception"}');

insert into public.sample_tag (sample_id, label)
select sample_id, 'Exemplar'
from public.anonymised_sample
where pseudo_student_id = 'S001';

insert into public.sample_tag (sample_id, label)
select sample_id, 'Misconception'
from public.anonymised_sample
where pseudo_student_id = 'S002';
