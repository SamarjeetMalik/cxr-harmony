"""Loading the canonical dataset into a queryable catalogue."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import create_engine, event, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url
from sqlalchemy.orm import Session, sessionmaker

from ..schema.models import CanonicalDataset
from ..workspace import Workspace
from .access import Role, require
from .schema_sql import (
    Base,
    InstanceRow,
    LabelRow,
    PatientRow,
    ReportRow,
    SeriesRow,
    SplitRow,
    StudyRow,
)


def _enable_foreign_keys(dbapi_connection, _record) -> None:
    """SQLite ignores foreign keys unless told otherwise, per connection."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def engine_url(target: Path | str) -> str:
    """Resolve a path or a URL to a SQLAlchemy URL.

    A ``str`` containing ``://`` is taken as a URL and passed through, so a
    deployment can point at PostgreSQL without editing code:

        CXR_HARMONY_DB=postgresql+psycopg://user@host/cxr_harmony

    Anything else is a filesystem path to a SQLite file, which is the demo
    default and stays the default.
    """
    if isinstance(target, str) and "://" in target:
        return target
    return f"sqlite:///{Path(target).as_posix()}"


def uses_sqlite(target: Path | str) -> bool:
    """Whether this target is SQLite, decided from the URL alone.

    Deliberately not ``create_engine(...).dialect.name``: that constructs the
    engine, which imports the DBAPI driver, so asking "is this SQLite?" about a
    PostgreSQL URL would fail on a machine with no ``psycopg`` installed. The
    question is answerable from the URL, so it is answered from the URL.
    """
    return make_url(engine_url(target)).get_backend_name() == "sqlite"


def open_engine(db_path: Path | str) -> Engine:
    """Open an engine against a SQLite file or any SQLAlchemy URL.

    Callers are responsible for ``dispose()``; prefer :func:`engine_scope`, which
    does it for them.

    The foreign-key pragma is attached only for SQLite, because it is a SQLite
    defect being worked around: PostgreSQL enforces declared foreign keys without
    being asked, and issuing the pragma there would fail on connect. This is the
    only dialect-specific behaviour in the catalogue, and
    ``tests/test_postgres_portability.py`` is what keeps it the only one.
    """
    engine = create_engine(engine_url(db_path), future=True)
    if uses_sqlite(db_path):
        event.listen(engine, "connect", _enable_foreign_keys)
    return engine


@contextmanager
def engine_scope(db_path: Path | str) -> Iterator[Engine]:
    """An engine that releases its file handle on exit.

    SQLAlchemy pools connections, so an undisposed engine holds the SQLite file
    open. On POSIX that is invisible — unlinking an open file succeeds — but on
    Windows it makes the file undeletable, so a rebuild of the catalogue fails
    with a permission error. Since CI runs on Linux, this is exactly the class of
    defect that ships green and breaks on a colleague's laptop.
    """
    engine = open_engine(db_path)
    try:
        yield engine
    finally:
        engine.dispose()


@dataclass
class CatalogStats:
    n_patients: int
    n_studies: int
    n_series: int
    n_instances: int
    n_reports: int
    n_labels: int
    per_site: dict[str, int]

    def to_dict(self) -> dict:
        return {
            "n_patients": self.n_patients,
            "n_studies": self.n_studies,
            "n_series": self.n_series,
            "n_instances": self.n_instances,
            "n_reports": self.n_reports,
            "n_labels": self.n_labels,
            "per_site": self.per_site,
        }


def build_catalog(dataset: CanonicalDataset, workspace: Workspace) -> CatalogStats:
    """Create the catalogue database from a canonical dataset, replacing any prior one."""
    workspace.ensure()
    if workspace.catalog_db.exists():
        workspace.catalog_db.unlink()

    with engine_scope(workspace.catalog_db) as engine:
        Base.metadata.create_all(engine)
        factory = sessionmaker(bind=engine, future=True)
        _populate(factory, dataset)
        return _read_stats(factory)


def _populate(factory, dataset: CanonicalDataset) -> None:
    with factory() as session:
        session.add_all(
            PatientRow(
                pseudo_id=p.pseudo_id,
                sex=p.sex.value,
                age_years=p.age_years,
            )
            for p in dataset.patients
        )
        session.flush()

        session.add_all(
            StudyRow(
                study_uid=s.study_uid,
                pseudo_patient_id=s.pseudo_patient_id,
                site_id=s.site_id,
                study_date=s.study_date,
                modality=s.modality,
                body_part=s.body_part,
            )
            for s in dataset.studies
        )
        session.flush()

        session.add_all(
            SeriesRow(
                series_uid=s.series_uid,
                study_uid=s.study_uid,
                view_position=s.view_position.value,
                laterality=s.laterality.value,
            )
            for s in dataset.series
        )
        session.flush()

        session.add_all(
            InstanceRow(
                sop_uid=i.sop_uid,
                series_uid=i.series_uid,
                relative_path=i.relative_path,
                sha256=i.sha256,
                rows=i.rows,
                columns=i.columns,
                bits_stored=i.bits_stored,
                photometric_interpretation=i.photometric_interpretation,
                pixel_redacted=i.pixel_redacted,
            )
            for i in dataset.instances
        )

        session.add_all(
            ReportRow(
                study_uid=r.study_uid,
                sections_json=json.dumps(
                    {k.value: v for k, v in r.sections.items()}, sort_keys=True
                ),
                redaction_count=r.redaction_count,
            )
            for r in dataset.reports
        )

        # A study may carry the same finding from two sources; the unique
        # constraint is on (study, finding, source), so de-duplicate within it.
        seen: set[tuple[str, str, str]] = set()
        for label in dataset.labels:
            key = (label.study_uid, label.finding.value, label.source.value)
            if key in seen:
                continue
            seen.add(key)
            session.add(
                LabelRow(
                    study_uid=label.study_uid,
                    finding=label.finding.value,
                    present=label.present,
                    source=label.source.value,
                    site_native_value=label.site_native_value,
                )
            )

        session.commit()


def _read_stats(factory) -> CatalogStats:
    with factory() as session:
        per_site = dict(
            session.execute(
                select(StudyRow.site_id, func.count(StudyRow.study_uid)).group_by(
                    StudyRow.site_id
                )
            ).all()
        )
        stats = CatalogStats(
            n_patients=session.scalar(select(func.count(PatientRow.pseudo_id))) or 0,
            n_studies=session.scalar(select(func.count(StudyRow.study_uid))) or 0,
            n_series=session.scalar(select(func.count(SeriesRow.series_uid))) or 0,
            n_instances=session.scalar(select(func.count(InstanceRow.sop_uid))) or 0,
            n_reports=session.scalar(select(func.count(ReportRow.study_uid))) or 0,
            n_labels=session.scalar(select(func.count(LabelRow.id))) or 0,
            per_site=per_site,
        )
    return stats


def record_splits(
    workspace: Workspace, assignments: dict[str, str], release_version: str
) -> int:
    """Persist a release's patient-level split assignment, replacing any prior one."""
    with engine_scope(workspace.catalog_db) as engine, Session(engine, future=True) as session:
        session.query(SplitRow).filter(SplitRow.release_version == release_version).delete()
        session.add_all(
            SplitRow(pseudo_patient_id=pid, release_version=release_version, split=split)
            for pid, split in sorted(assignments.items())
        )
        session.commit()
    return len(assignments)


def query_studies(workspace: Workspace, role: Role, *, site_id: str | None = None) -> list[dict]:
    """Row-level study listing, subject to ``role``."""
    require(role, "studies")
    with engine_scope(workspace.catalog_db) as engine, Session(engine, future=True) as session:
        stmt = select(StudyRow)
        if site_id:
            stmt = stmt.where(StudyRow.site_id == site_id)
        return [
            {
                "study_uid": row.study_uid,
                "pseudo_patient_id": row.pseudo_patient_id,
                "site_id": row.site_id,
                "study_date": row.study_date.isoformat() if row.study_date else None,
                "modality": row.modality,
            }
            for row in session.scalars(stmt)
        ]


def query_report_text(workspace: Workspace, role: Role, study_uid: str) -> dict:
    """Report sections for one study. Curator only."""
    require(role, "reports")
    with engine_scope(workspace.catalog_db) as engine, Session(engine, future=True) as session:
        row = session.get(ReportRow, study_uid)
        return json.loads(row.sections_json) if row is not None else {}


def summary_counts(workspace: Workspace, role: Role) -> dict:
    """Aggregates only. Available to every role, including the auditor.

    ``role`` is accepted but not checked: oversight must never require access to
    the rows being overseen, so aggregates are open to all three roles.
    """
    _ = role
    with engine_scope(workspace.catalog_db) as engine, Session(engine, future=True) as session:
        return {
            "patients": session.scalar(select(func.count(PatientRow.pseudo_id))) or 0,
            "studies": session.scalar(select(func.count(StudyRow.study_uid))) or 0,
            "instances": session.scalar(select(func.count(InstanceRow.sop_uid))) or 0,
            "per_site": dict(
                session.execute(
                    select(StudyRow.site_id, func.count(StudyRow.study_uid)).group_by(
                        StudyRow.site_id
                    )
                ).all()
            ),
        }


__all__ = [
    "CatalogStats",
    "build_catalog",
    "engine_scope",
    "open_engine",
    "query_report_text",
    "query_studies",
    "record_splits",
    "summary_counts",
]
