# Deployment

What this repository ships is a **demonstration configuration**. It runs on a
laptop with no server, which is what makes it evaluable in forty seconds and is
a deliberate property, not an oversight. This document is the difference between
that and a deployment, written out so the gap is explicit rather than assumed
closed.

---

## What the demo configuration is not

| | Demo (this repo) | Deployment |
|---|---|---|
| Catalogue | SQLite file | PostgreSQL 15 with row-level security |
| Object storage | Local filesystem | S3-compatible (MinIO), catalogue holds metadata only |
| Access control | Application-level, in the query helpers | Database roles and RLS policies |
| Pseudonymisation key | File in the working directory | KMS or HSM, or a mounted secret |
| Encryption at rest | None | Encrypted volume; TDE on PostgreSQL |
| Concurrency | Single process | Multiple workers against a shared catalogue |

The migration is not attempted *in the demo path*, and that has not changed:
`make demo` still runs on a laptop with no server, which is what makes this
repository evaluable in forty seconds.

What has changed is that the deployment configuration is now startable rather
than only readable. [`deploy/docker-compose.yml`](../deploy/docker-compose.yml)
brings up PostgreSQL and MinIO, and
[`deploy/postgres/01-roles-and-rls.sql`](../deploy/postgres/01-roles-and-rls.sql)
applies the roles and row-level-security policies described below. It is
additive: nothing in the demo path requires it.

**What is verified, and what is not.** `tests/test_postgres_portability.py`
checks that the catalogue schema compiles as valid PostgreSQL DDL with all six
foreign keys intact, that the SQLite-only foreign-key pragma is not attached to a
PostgreSQL engine, and that the policy file references only tables and columns
the schema actually has. None of that is the same as running it. The compose
stack has never been brought up — there is no Docker on the machine it was
written on and no server in CI — so the policies have not been applied to a live
database, and the MinIO service is configuration for an object-store backend that
remains unimplemented. This is the intended production shape, not a demonstrated
one, and the compose file says so in its own header.

---

## Pseudonymisation key

This is the one control where the demo default is genuinely unsafe in production,
so the code refuses to make the unsafe choice silently.

`load_or_create_key` **raises** rather than creating a key, unless
`allow_create=True` is passed. When it does create one it emits a
`RuntimeWarning`. Silent creation is how a deployment ends up with the secret
that reverses every pseudonym sitting in a working directory, inside whatever
backs that directory up — and because the pipeline would run perfectly, nobody
finds out until an audit.

**In deployment**, supply the key through the environment:

```bash
export CXR_HARMONY_KEY=$(openssl rand -hex 32)
```

```python
from cxr_harmony.deid import Pseudonymiser
pseudo = Pseudonymiser.from_env()
```

A secret mounted by Docker or Kubernetes never touches the filesystem, so it
cannot be picked up by a backup, baked into an image layer, or left behind on a
decommissioned volume.

```yaml
# docker-compose
services:
  pipeline:
    image: cxr-harmony:latest
    environment:
      CXR_HARMONY_KEY_FILE: /run/secrets/cxr_key
    secrets: [cxr_key]
secrets:
  cxr_key:
    external: true
```

**Key destruction is the event that makes the cohort irreversibly anonymous.** It
should be a dated, documented decision with sign-off, not a side effect of a
machine being reimaged. Until it happens, the release remains personal data — see
[`dpdp-controls.md`](dpdp-controls.md).

---

## Catalogue: moving to PostgreSQL

The catalogue is SQLAlchemy over SQLite. The models are backend-agnostic; the
engine URL is the only thing that is not:

```python
# src/cxr_harmony/catalog/store.py
create_engine("postgresql+psycopg://user@host/cxr_harmony")
```

SQLite is the right default for a demonstration and the wrong one at scale: it
serialises writers, has no row-level security, and cannot shard. At the million-
study scale this pipeline is designed toward, the object store is the throughput
bottleneck rather than the catalogue — but the *security* argument for moving is
independent of scale and applies immediately.

### Row-level security

Application-level roles (`cxr_harmony.catalog.access`) constrain the query
helpers. They do not constrain anyone holding the database file, and the module
says so. In PostgreSQL the same policy is enforced by the database:

```sql
ALTER TABLE reports ENABLE ROW LEVEL SECURITY;

-- Report prose is the highest residual re-identification surface after
-- scrubbing, so the population with routine access to it is kept small.
CREATE POLICY reports_curator_only ON reports
    FOR SELECT TO curator USING (true);
CREATE POLICY reports_no_modeller ON reports
    FOR SELECT TO modeller USING (false);

-- The auditor sees aggregates, never patient-level rows. Oversight must not
-- require access to the thing being overseen.
REVOKE SELECT ON studies, series, instances, reports FROM auditor;
GRANT SELECT ON cohort_summary TO auditor;
```

### Encryption at rest

- **PostgreSQL**: transparent data encryption, or an encrypted volume.
- **SQLite in demo mode**: SQLCipher if the working directory is not itself on an
  encrypted volume.
- **In all cases**: the working directory holds the pseudonymisation key and the
  de-identified store. It must live on an encrypted volume with access restricted
  to the service account, and it must not be included in any backup that leaves
  the secure zone.

---

## Object storage

Ingest indexes in place and copies no pixels; the only image bytes written are the
de-identified ones under `work/deid`. At deployment scale that directory becomes a
bucket:

```
s3://cohort/deid/<site>/<study_uid>/<sop_uid>.dcm
```

The catalogue holds paths and digests, never pixels, so the change is confined to
how `relative_path` is resolved. Release manifests already carry a SHA-256 per
file, so integrity across the move is verifiable with the tooling that exists.

---

## Cross-border transfer

DPDP s.16 bears directly on any collaboration that moves data outside India. The
release manifest identifies exactly which files a recipient holds, which is what
makes a destruction request actionable — but the decision about what may leave
the country is contractual and belongs in the data-sharing agreement, not in
code. See [`dpdp-controls.md`](dpdp-controls.md).

---

## Not implemented

Named so the gap is explicit:

- **PostgreSQL migration, MinIO integration, RLS policies** — documented above,
  not implemented.
- **HL7v2 listener** — a listener is a server, and shipping half of one is worse
  than shipping none.
- **DICOM-SR parsing** — the FHIR export covers the outbound direction; inbound
  structured reports are not parsed.
- **Body-part classifier** — the gate exists (`BODY_PART_UNVERIFIED`, and a
  release-blocking QC check); the model does not, because no labelled body-part
  data was reachable. See [`RESULTS.md`](RESULTS.md).
