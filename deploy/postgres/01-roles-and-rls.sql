-- Row-level security for the catalogue, as a runnable file rather than prose.
--
-- docs/deployment.md has described these policies since the first publish. They
-- were correct SQL in a Markdown fence, which is not the same as SQL anyone can
-- run: nothing checked that the table and column names still matched the schema,
-- and a policy that references a renamed column fails at apply time, in
-- production, during a migration.
--
-- Applied automatically by deploy/docker-compose.yml, which mounts this
-- directory into the PostgreSQL image's entrypoint.
--
-- SCOPE, stated because the rest of this repository states its scope: the
-- schema these policies attach to is verified to compile under PostgreSQL by
-- tests/test_postgres_portability.py. The *policies themselves* have not been
-- executed against a live server by this project, because there is no server in
-- CI and no Docker on the machine this was written on. They are the intended
-- configuration, not a demonstrated one.

-- --------------------------------------------------------------------------
-- Roles. Mirrors cxr_harmony.catalog.access, which enforces the same split at
-- application level. Two layers, because the application layer constrains the
-- query helpers and this one constrains anyone holding a connection.
-- --------------------------------------------------------------------------

DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'curator') THEN
        CREATE ROLE curator;
    END IF;
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'modeller') THEN
        CREATE ROLE modeller;
    END IF;
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'auditor') THEN
        CREATE ROLE auditor;
    END IF;
END
$$;

GRANT CONNECT ON DATABASE cxr_harmony TO curator, modeller, auditor;
GRANT USAGE ON SCHEMA public TO curator, modeller, auditor;

-- --------------------------------------------------------------------------
-- Report prose is the highest residual re-identification surface after
-- scrubbing: free text carries indirect identifiers that no tag-level profile
-- can catch. So the population with routine access to it is kept small.
-- --------------------------------------------------------------------------

ALTER TABLE reports ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS reports_curator_only ON reports;
CREATE POLICY reports_curator_only ON reports
    FOR SELECT TO curator USING (true);

DROP POLICY IF EXISTS reports_no_modeller ON reports;
CREATE POLICY reports_no_modeller ON reports
    FOR SELECT TO modeller USING (false);

-- --------------------------------------------------------------------------
-- The modeller sees images and labels, never prose. That is the whole point of
-- separating the roles: a training run does not need the report it was
-- derived from.
-- --------------------------------------------------------------------------

GRANT SELECT ON patients, studies, series, instances, labels, splits TO modeller;
GRANT SELECT ON patients, studies, series, instances, labels, splits, reports TO curator;

-- --------------------------------------------------------------------------
-- Oversight must not require access to the thing being overseen. The auditor
-- sees aggregates and never patient-level rows.
-- --------------------------------------------------------------------------

CREATE OR REPLACE VIEW cohort_summary AS
    SELECT
        s.site_id,
        p.sex,
        COUNT(DISTINCT s.study_uid)        AS n_studies,
        COUNT(DISTINCT s.pseudo_patient_id) AS n_patients
    FROM studies s
    JOIN patients p ON p.pseudo_id = s.pseudo_patient_id
    GROUP BY s.site_id, p.sex;

REVOKE ALL ON patients, studies, series, instances, reports, labels, splits FROM auditor;
GRANT SELECT ON cohort_summary TO auditor;
