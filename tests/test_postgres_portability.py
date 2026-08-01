"""Check the PostgreSQL claim the catalogue has been making all along.

``schema_sql.py`` says the models "point at PostgreSQL by changing a URL when a
deployment outgrows it", and ``docs/deployment.md`` builds a migration plan on
top of that. Nothing checked it. It was the same shape as the four decomposition
figures that turned out to be unreproducible: a plausible claim, written once,
never exercised.

It can be checked without a server. SQLAlchemy will compile DDL against any
dialect through a mock engine, which is enough to catch the things that actually
break a port — a column type with no PostgreSQL equivalent, a reserved word used
as an identifier, a SQLite-only construct.

**What this does not prove.** Compiling DDL is not running a deployment. It says
nothing about connection handling, transaction semantics, concurrent writers, or
whether the row-level-security policies in ``docs/deployment.md`` behave as
described. Those need a live server, and this repository has no way to stand one
up in CI without becoming the thing ``deployment.md`` argues against. The claim
verified here is exactly "the schema is portable", no wider.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from sqlalchemy import create_mock_engine

from cxr_harmony.catalog.schema_sql import Base
from cxr_harmony.catalog.store import engine_url, open_engine, uses_sqlite

DIALECTS = ["postgresql://", "sqlite://", "mysql://"]


def _ddl_for(dialect_url: str) -> list[str]:
    statements: list[str] = []

    def collect(sql, *args, **kwargs):
        statements.append(str(sql.compile(dialect=engine.dialect)))

    engine = create_mock_engine(dialect_url, collect)
    Base.metadata.create_all(engine, checkfirst=False)
    return statements


@pytest.mark.parametrize("dialect_url", DIALECTS, ids=lambda u: u.split(":")[0])
def test_schema_compiles_under_postgresql(dialect_url):
    """Every table must render as valid DDL for each backend.

    MySQL is included not because anyone plans to use it, but because a second
    non-SQLite dialect is what distinguishes "portable" from "happens to work on
    the one other backend we tried".
    """
    statements = _ddl_for(dialect_url)
    assert statements, f"no DDL emitted for {dialect_url}"

    tables = {t.name for t in Base.metadata.sorted_tables}
    rendered = "\n".join(statements)
    for table in tables:
        assert f"CREATE TABLE {table}" in rendered, (
            f"{table} did not render under {dialect_url}"
        )


def test_foreign_keys_survive_the_port():
    """The catalogue's integrity guarantee must not be SQLite-only.

    SQLite needs a pragma to enforce foreign keys at all, which the store
    attaches on connect. If the *declarations* were lost in the PostgreSQL DDL,
    the port would silently drop the constraint that stops the catalogue filling
    with orphaned rows.
    """
    rendered = "\n".join(_ddl_for("postgresql://"))
    expected = sum(len(t.foreign_keys) for t in Base.metadata.sorted_tables)
    assert expected > 0, "no foreign keys declared; this test is watching nothing"
    assert rendered.count("FOREIGN KEY") == expected, (
        f"expected {expected} foreign keys in the PostgreSQL DDL, "
        f"found {rendered.count('FOREIGN KEY')}"
    )


@pytest.mark.parametrize(
    "url",
    [
        "postgresql+psycopg://user@host/db",
        "postgresql://user@host/db",
        "mysql+pymysql://user@host/db",
    ],
)
def test_the_sqlite_pragma_is_not_attached_to_other_backends(url):
    """`PRAGMA foreign_keys` is a SQLite workaround and fails elsewhere.

    Attaching it unconditionally is the obvious way to break the port, and it
    would not show up in a DDL comparison — the failure happens on connect.

    Asked of the URL rather than of a constructed engine, because constructing
    one imports the DBAPI driver: this must be answerable on a laptop with no
    `psycopg` installed, which is every laptop running the demo.
    """
    assert not uses_sqlite(url)


def test_a_sqlite_target_still_gets_the_pragma(tmp_path):
    """Guards the test above from passing because the check inverted."""
    assert uses_sqlite(tmp_path / "catalog.db")
    assert uses_sqlite("sqlite:///:memory:")

    engine = open_engine("sqlite://")
    try:
        listeners = getattr(engine.pool.dispatch, "connect", [])
        assert any(
            getattr(fn, "__name__", "") == "_enable_foreign_keys" for fn in listeners
        ), "SQLite lost its foreign-key pragma, so constraints are unenforced"
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    "target,expected",
    [
        ("postgresql+psycopg://u@h/db", "postgresql+psycopg://u@h/db"),
        ("sqlite:///:memory:", "sqlite:///:memory:"),
    ],
)
def test_urls_pass_through_unchanged(target, expected):
    assert engine_url(target) == expected


def test_a_path_becomes_a_sqlite_url(tmp_path):
    """The demo default. A bare path must not be mistaken for a URL."""
    assert engine_url(tmp_path / "catalog.db").startswith("sqlite:///")


# --- the RLS policies, checked against the schema they attach to -------------

RLS_SQL = (
    Path(__file__).resolve().parents[1] / "deploy" / "postgres" / "01-roles-and-rls.sql"
)


def _schema_tables() -> set[str]:
    return {t.name for t in Base.metadata.sorted_tables}


def test_rls_policies_reference_tables_that_exist():
    """A policy naming a renamed table fails at apply time, in production.

    These policies lived in a Markdown fence in docs/deployment.md, where nothing
    could check them against the schema. Moving them into a real file is only an
    improvement if something now does.
    """
    sql = RLS_SQL.read_text(encoding="utf-8")
    known = _schema_tables() | {"cohort_summary"}

    referenced: set[str] = set()
    for statement in re.findall(
        r"\b(?:ALTER TABLE|GRANT SELECT ON|REVOKE ALL ON)\s+"
        r"([a-z_,\s]+?)(?:\s+(?:TO|FROM|ENABLE)\b)",
        sql,
    ):
        referenced.update(name.strip() for name in statement.split(",") if name.strip())

    assert referenced, "no table references found; this test is watching nothing"
    unknown = sorted(referenced - known)
    assert not unknown, (
        f"the RLS policies reference tables that are not in the schema: {unknown}. "
        f"Schema has: {sorted(known)}"
    )


def test_the_audit_view_uses_columns_that_exist():
    """`cohort_summary` is what the auditor sees instead of patient rows.

    It joins two tables by hand, so it is the part of this file most likely to
    drift when a column is renamed — and the part whose failure would revoke
    oversight rather than grant too much.
    """
    sql = RLS_SQL.read_text(encoding="utf-8")
    columns = {c.name for t in Base.metadata.sorted_tables for c in t.columns}
    for column in ("site_id", "sex", "study_uid", "pseudo_patient_id", "pseudo_id"):
        assert column in columns, f"{column} is no longer in the schema"
        assert column in sql, f"cohort_summary no longer references {column}"


def test_the_policies_do_not_grant_reports_to_the_modeller():
    """The one substantive rule: report prose is curator-only.

    Asserted as absence, not presence — a GRANT that appears anywhere is a leak
    regardless of what other policies say, and the ordering of DROP/CREATE would
    make a presence check pass while the wrong grant sat below it.
    """
    sql = RLS_SQL.read_text(encoding="utf-8")
    for grant in re.findall(r"GRANT SELECT ON\s+([a-z_,\s]+?)\s+TO\s+([a-z_,\s]+?);", sql):
        tables = {t.strip() for t in grant[0].split(",")}
        roles = {r.strip() for r in grant[1].split(",")}
        if "reports" in tables:
            assert "modeller" not in roles, (
                "the modeller has been granted SELECT on reports; free-text prose "
                "is the highest residual re-identification surface after scrubbing"
            )
