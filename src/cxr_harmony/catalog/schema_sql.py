"""Relational schema for the catalogue.

SQLite rather than PostgreSQL, deliberately. The catalogue has to be openable by
whoever picks this up next, on a laptop, with no server to provision — and at
cohort sizes in the low millions of rows SQLite is not the bottleneck; the object
store is. The SQLAlchemy layer means the same models point at PostgreSQL by
changing a URL when a deployment outgrows it.

Every foreign key is declared and enforced. SQLite does not enforce them unless
asked, which is a well-known way to end up with a catalogue full of orphaned rows
that nobody notices until a training run silently drops a third of its studies.
"""

from __future__ import annotations

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class PatientRow(Base):
    __tablename__ = "patients"

    pseudo_id = Column(String(16), primary_key=True)
    sex = Column(String(8), nullable=False, default="U")
    age_years = Column(Integer, nullable=True)

    studies = relationship("StudyRow", back_populates="patient")


class StudyRow(Base):
    __tablename__ = "studies"

    study_uid = Column(String(64), primary_key=True)
    pseudo_patient_id = Column(
        String(16), ForeignKey("patients.pseudo_id"), nullable=False, index=True
    )
    site_id = Column(String(16), nullable=False, index=True)
    study_date = Column(Date, nullable=True)
    modality = Column(String(16), nullable=False)
    body_part = Column(String(32), nullable=True)

    patient = relationship("PatientRow", back_populates="studies")
    series = relationship("SeriesRow", back_populates="study")


class SeriesRow(Base):
    __tablename__ = "series"

    series_uid = Column(String(64), primary_key=True)
    study_uid = Column(String(64), ForeignKey("studies.study_uid"), nullable=False, index=True)
    view_position = Column(String(24), nullable=False, default="UNKNOWN")
    laterality = Column(String(8), nullable=False, default="NA")

    study = relationship("StudyRow", back_populates="series")


class InstanceRow(Base):
    __tablename__ = "instances"

    sop_uid = Column(String(64), primary_key=True)
    series_uid = Column(String(64), ForeignKey("series.series_uid"), nullable=False, index=True)
    relative_path = Column(String(255), nullable=False)
    sha256 = Column(String(64), nullable=False, index=True)
    rows = Column(Integer, nullable=False)
    columns = Column(Integer, nullable=False)
    bits_stored = Column(Integer, nullable=False)
    photometric_interpretation = Column(String(24), nullable=False)
    pixel_redacted = Column(Boolean, nullable=False, default=False)


class ReportRow(Base):
    __tablename__ = "reports"

    study_uid = Column(String(64), ForeignKey("studies.study_uid"), primary_key=True)
    #: Sections are stored as one JSON blob rather than a column each, because
    #: which sections a site emits varies and a schema migration per house style
    #: is not a sustainable arrangement.
    sections_json = Column(Text, nullable=False, default="{}")
    redaction_count = Column(Integer, nullable=False, default=0)


class LabelRow(Base):
    __tablename__ = "labels"

    id = Column(Integer, primary_key=True, autoincrement=True)
    study_uid = Column(String(64), ForeignKey("studies.study_uid"), nullable=False, index=True)
    finding = Column(String(32), nullable=False, index=True)
    present = Column(Boolean, nullable=False, default=True)
    source = Column(String(24), nullable=False)
    site_native_value = Column(String(64), nullable=True)

    __table_args__ = (
        UniqueConstraint("study_uid", "finding", "source", name="uq_label_per_source"),
    )


class SplitRow(Base):
    __tablename__ = "splits"

    pseudo_patient_id = Column(String(16), ForeignKey("patients.pseudo_id"), primary_key=True)
    release_version = Column(String(32), primary_key=True)
    split = Column(String(8), nullable=False, index=True)


__all__ = [
    "Base",
    "InstanceRow",
    "LabelRow",
    "PatientRow",
    "ReportRow",
    "SeriesRow",
    "SplitRow",
    "StudyRow",
]
