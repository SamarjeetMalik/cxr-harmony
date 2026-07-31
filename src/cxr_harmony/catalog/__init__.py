"""Queryable catalogue over the canonical dataset, with role-based access."""

from .access import ROW_ACCESS, AccessDenied, Role, require
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
from .store import (
    CatalogStats,
    build_catalog,
    engine_scope,
    open_engine,
    query_report_text,
    query_studies,
    record_splits,
    summary_counts,
)

__all__ = [
    "ROW_ACCESS",
    "AccessDenied",
    "Base",
    "CatalogStats",
    "InstanceRow",
    "LabelRow",
    "PatientRow",
    "ReportRow",
    "Role",
    "SeriesRow",
    "SplitRow",
    "StudyRow",
    "build_catalog",
    "engine_scope",
    "open_engine",
    "query_report_text",
    "query_studies",
    "record_splits",
    "require",
    "summary_counts",
]
