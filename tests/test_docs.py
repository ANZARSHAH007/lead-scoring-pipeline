"""The docs are checked by the suite, not by eye (task T-703).

Reconciling docs 01-14 against the code on 2026-11-21 found three things that had drifted
silently: docs/06 described a schema with four `CHECK` constraints and a column called
`reported_at` that the code never had, docs/07 had a misplaced code fence that rendered a
paragraph as JSON, and the README still said "the pipeline itself is not built yet".

Fixing those three is worth an afternoon. Stopping them recurring is worth more, and that is
what this file is: the parts of the documentation that *can* be checked mechanically now are,
so the next drift fails a test instead of waiting for someone to read carefully.

What is deliberately not here: prose. No test can tell whether an explanation is still true.
These cover the shapes - endpoints, schemas, examples - where the code is the authority.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from leadscore.api import main as api_main
from leadscore.api import schemas as sc
from leadscore.api import store

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

API_DOC = DOCS / "07_API_DOCUMENTATION.md"
DB_DOC = DOCS / "06_DATABASE_DESIGN.md"


def json_blocks(text: str) -> list[str]:
    return re.findall(r"```json\s*(.*?)```", text, re.S)


def create_tables(text: str) -> dict[str, list[str]]:
    """Every `CREATE TABLE ... (...)` as {name: [column definitions]}, comments stripped."""
    tables: dict[str, list[str]] = {}
    for match in re.finditer(r"CREATE TABLE IF NOT EXISTS (\w+)\s*\((.*?)\);", text, re.S):
        columns = []
        for line in match.group(2).splitlines():
            line = re.sub(r"--.*$", "", line).strip().rstrip(",").strip()
            if line:
                columns.append(re.sub(r"\s+", " ", line))
        tables[match.group(1)] = columns
    return tables


# --- docs/07: the API documentation -----------------------------------------------------


def test_every_json_example_in_the_api_doc_is_valid_json() -> None:
    """A misplaced code fence makes prose render as JSON. That is how this one was found."""
    blocks = json_blocks(API_DOC.read_text(encoding="utf-8"))
    assert blocks, "docs/07 should contain JSON examples"
    for index, raw in enumerate(blocks):
        try:
            json.loads(raw)
        except ValueError as error:
            pytest.fail(f"JSON example {index + 1} in docs/07 does not parse: {error}")


def test_the_documented_endpoints_are_the_endpoints_that_exist() -> None:
    documented = set(re.findall(r"### (?:GET|POST) `(/[\w/-]*)`", API_DOC.read_text("utf-8")))
    real = {
        route.path
        for route in api_main.app.routes
        if getattr(route, "path", "").startswith("/")
        and route.path not in {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}
    }
    assert documented == real, (
        f"docs/07 documents {sorted(documented)} but the app serves {sorted(real)}"
    )


def test_the_documented_score_request_validates_against_the_real_model() -> None:
    """The example a reader will copy must be one the API actually accepts."""
    match = re.search(
        r"### POST `/score`.*?\*\*Request\*\*\s*```json\s*(.*?)```",
        API_DOC.read_text(encoding="utf-8"),
        re.S,
    )
    assert match, "docs/07 no longer shows a request example for POST /score"
    lead = sc.Lead(**json.loads(match.group(1)))
    assert lead.prospect_id


def test_the_documented_request_example_uses_every_field_of_the_model() -> None:
    """An example that quietly omits fields teaches readers an incomplete shape."""
    match = re.search(
        r"### POST `/score`.*?\*\*Request\*\*\s*```json\s*(.*?)```",
        API_DOC.read_text(encoding="utf-8"),
        re.S,
    )
    assert match
    assert set(json.loads(match.group(1))) == set(sc.Lead.model_fields)


# --- docs/06: the database design -------------------------------------------------------


def test_the_documented_schema_is_the_schema_the_code_creates() -> None:
    """docs/06 drifted from `store.py` for two phases before anyone noticed (T-703)."""
    documented = create_tables(DB_DOC.read_text(encoding="utf-8"))
    real = create_tables(store.SCHEMA)
    assert set(documented) == set(real), "docs/06 and store.py disagree on which tables exist"
    for table in sorted(real):
        assert documented[table] == real[table], (
            f"docs/06's `{table}` no longer matches store.py. Copy the SQL from store.py, "
            f"which is where the schema is actually defined."
        )


# --- the README: the one claim a stranger reads first ------------------------------------


def test_the_readme_does_not_claim_the_pipeline_is_unbuilt() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "not built yet" not in readme, (
        "the README's status line is stale again - it survived five phases of being wrong"
    )


def test_the_readme_links_its_ci_badge_at_this_repository() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "actions/workflows/ci.yml/badge.svg" in readme, "the CI badge is missing"
