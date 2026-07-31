"""Fabricated identifier pools used to populate the synthetic corpus.

Everything here is invented. The point of generating realistic-looking direct
identifiers is that the de-identification stage then has something real to
remove: running a de-identifier over a corpus that was already de-identified
demonstrates nothing.

The national health identifier deserves a note. Indian partner sites increasingly
record an ABHA (Ayushman Bharat Health Account) number alongside their local MRN.
Where two hospitals both record it for the same person, it is the only field that
can link them — local MRNs cannot, by construction. The generator therefore emits
a controlled amount of cross-site patient overlap carrying a shared national ID,
so that :mod:`cxr_harmony.deid` can be shown collapsing those records to a single
pseudonym. The format below mimics the real 14-digit layout but every value is
drawn from a reserved test prefix that cannot collide with a live account.
"""

from __future__ import annotations

import random

#: Reserved leading digits so no generated value can resemble a live ABHA number.
ABHA_TEST_PREFIX = "99"

SURNAMES = (
    "Sharma", "Verma", "Nair", "Reddy", "Iyer", "Banerjee", "Chatterjee", "Patel",
    "Desai", "Kulkarni", "Joshi", "Mishra", "Pillai", "Menon", "Bhattacharya",
    "Gogoi", "Baruah", "Mahanta", "Das", "Sen", "Ghosh", "Rao", "Naidu", "Shetty",
    "Hegde", "Kaur", "Singh", "Gill", "Chauhan", "Rathore", "Mehta", "Shah",
    "Trivedi", "Panda", "Mohapatra", "Sahoo", "Behera", "Jena", "Dutta", "Bose",
)

MALE_GIVEN_NAMES = (
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Reyansh", "Ayaan", "Krishna",
    "Ishaan", "Shaurya", "Rohan", "Kabir", "Rajesh", "Suresh", "Ramesh", "Anil",
    "Deepak", "Manoj", "Vikram", "Sanjay", "Arun", "Nikhil", "Pranav", "Siddharth",
)

FEMALE_GIVEN_NAMES = (
    "Ananya", "Diya", "Aadhya", "Saanvi", "Pari", "Anika", "Navya", "Myra",
    "Kiara", "Riya", "Devika", "Meera", "Lakshmi", "Sunita", "Pooja", "Kavita",
    "Neha", "Priya", "Sneha", "Ishita", "Tanvi", "Shreya", "Aditi", "Nandini",
)

#: Used where the person's sex is irrelevant, such as clinician names.
GIVEN_NAMES = MALE_GIVEN_NAMES + FEMALE_GIVEN_NAMES

PHYSICIAN_SURNAMES = (
    "Krishnan", "Ramachandran", "Sundaram", "Venkatesh", "Bhandari", "Chopra",
    "Malhotra", "Saxena", "Tiwari", "Dubey", "Kapoor", "Ahluwalia", "Fernandes",
    "D'Souza", "Sequeira", "Thomas", "Varghese", "Kurian", "Mathew", "Abraham",
)

STREETS = (
    "MG Road", "Station Road", "Nehru Nagar", "Gandhi Chowk", "Civil Lines",
    "Model Town", "Sector 14", "Ring Road", "Park Street", "Church Street",
)


def make_person_name(rng: random.Random, sex: str = "U") -> tuple[str, str]:
    """Return ``(dicom_form, display_form)`` for a fabricated patient name.

    The given name is drawn to agree with ``sex``. A corpus in which names and
    sex codes disagree looks synthetic at a glance, and the point of this data is
    to be a fair test of tools that will later meet real records.
    """
    surname = rng.choice(SURNAMES)
    pool = {"M": MALE_GIVEN_NAMES, "F": FEMALE_GIVEN_NAMES}.get(sex, GIVEN_NAMES)
    given = rng.choice(pool)
    return f"{surname.upper()}^{given.upper()}", f"{given} {surname}"


def make_physician_name(rng: random.Random) -> tuple[str, str]:
    """Return ``(dicom_form, display_form)`` for a fabricated physician name."""
    surname = rng.choice(PHYSICIAN_SURNAMES)
    given = rng.choice(GIVEN_NAMES)
    return f"{surname.upper()}^{given.upper()}^^Dr", f"Dr. {given} {surname}"


def make_abha_number(rng: random.Random) -> str:
    """A 14-digit national health identifier in the conventional 2-4-4-4 grouping."""
    digits = ABHA_TEST_PREFIX + "".join(str(rng.randint(0, 9)) for _ in range(12))
    return f"{digits[0:2]}-{digits[2:6]}-{digits[6:10]}-{digits[10:14]}"


def make_phone(rng: random.Random) -> str:
    """A fabricated mobile number using the reserved 99xxx test range."""
    return "+91-99" + "".join(str(rng.randint(0, 9)) for _ in range(8))


def make_address(rng: random.Random, city: str) -> str:
    return f"{rng.randint(1, 240)}, {rng.choice(STREETS)}, {city}"
