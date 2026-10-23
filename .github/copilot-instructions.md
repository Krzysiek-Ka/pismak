# Tests

Three kinds of tests, never mixed:

| Kind | Folder | Scope | Model and DB | When it runs |
|---|---|---|---|---|
| unit | `tests/unit_test/` | one function | mocked | every `pytest` |
| golden | `tests/golden_set/<module>/` | one module | real | manually: `pytest -m golden` |
| e2e | `tests/e2e/` | whole pipeline | real, no mocks | manually: `pytest -m e2e` |

## General rules

- Use pytest only. Mock with `unittest.mock` and `monkeypatch`. Do not add test dependencies without asking.
- First read the code under test, the existing tests and the `conftest.py` files. Reuse fixtures.
- Run the tests you wrote and report the command and the result. Never claim a test passes without running it.
- If the code is wrong, report the bug. Do not change the expected values.
- Never use real secrets. In unit tests `OPENROUTER_KEY` is a dummy value set with `monkeypatch.setenv`. Elsewhere it is read from `.env`. Never log it.

## Unit tests

- Mirror `src/`: `src/Zrodlowyszukiwator/rag_ingestion/repository.py` → `tests/unit_test/Zrodlowyszukiwator/rag_ingestion/test_repository.py`.
- Test folders reuse package names from `src/`, so every folder under `tests/` (including `tests/`) needs an `__init__.py`, unless the project runs pytest with `--import-mode=importlib`.
- One class per function: `Test<FunctionName>` (`TestGenerateStructured`; for a method, `TestPgVectorRepositoryEnsureSchema`). Test name: `test_<function_name>__<scenario>_<expected_result>`.
- Each test has a short docstring: which scenario it tests and why it matters.
- No network, LLM, Ollama, PostgreSQL, or files outside `tmp_path`. Mock at the boundary (LLM client, DB connection, embedding model), not private helpers.
- Assert on behavior visible from outside: the return value, `pytest.raises(..., match=...)`, the arguments passed to the mocked client. One scenario per test; variants go in `parametrize` with readable `ids`.
- Code that calls an LLM (e.g. `generate_structured` in `src/_shared/serwer.py`) must be tested for: valid JSON; JSON in a Markdown fence; an empty or non-JSON response; a Pydantic validation error; a response cut off by the token limit; a timeout and a provider error; an explicit token limit from the module constant on every call, including retries; and a bounded number of retries that ends in a readable error.

## Golden set

```text
tests/golden_set/<module_name>/     # Dopytywator, Zrodlowyszukiwator, Pismogenerator
├── test_golden_<module_name>.py
├── cases/<case_id>.json            # one file per case
└── results/                        # run reports, ignored by Git
```

- Each test module starts with `pytestmark = pytest.mark.golden`.
- The model, DB, retriever, prompts and schemas are real. The Dopytywator gets its user answers from the case file, in order. Only parts outside the pipeline (e.g. the UI) may be faked, with a comment saying why.
- Parametrize over providers/models from the project config, never hardcoded. Test id: `<provider>-<case_id>`.
- If a provider or the DB is unreachable before the test starts, call `pytest.skip("<reason>")`.
- Always set limits: the module's token limit, a timeout per call, and `max_questions` from the case.

All case data lives in the case file, none of it in Python:

```json
{
  "id": "buspas_kazimierza_wielkiego",
  "description": "What the case checks and why (acts as the docstring).",
  "input": "Chciałbym wprowadzenia buspasa na ulicy Kazimierza Wielkiego we Wrocławiu.",
  "user_answers": ["Są tam korki, przez które komunikacja miejska jedzie wolniej."],
  "limits": { "max_questions": 12 },
  "expected": { "must_mention": ["Kazimierza Wielkiego"], "expected_source_ids": [], "expect_no_relevant_source": false },
  "manual_review": ["The brief does not present the user's observations as verified facts."]
}
```

- `expected` holds checkable properties, never exact text. Schema validation is always checked. The test file lists which `expected` fields it supports.
- Anything that cannot be checked mechanically goes to `manual_review`. It is copied into the report and never fails the test.
- Each run writes `results/<timestamp>_<provider>_<case_id>.json` with: the exact model name, `status` (`passed` / `failed_check` / `infra_error` / `skipped`), the result of each check, the number of questions and why the conversation stopped, the returned sources, duration, retries, errors and `manual_review`. No secrets.
- **Zrodlowyszukiwator:** 3 cases per corpus document: one answered directly by the document, one answered by combining information from it, and one not covered by the corpus (`expect_no_relevant_source: true`). Check that the right document is returned.
- **Dopytywator:** it ends the conversation once it has enough information and never exceeds `max_questions`. Do not require an exact number of questions.
- **Pismogenerator:** no facts, measurements, quotes or legal bases that are not in the brief or the sources.

## E2E

Same rules as the golden set (case files, providers, skip, limits, report, `manual_review`), with these differences:

- Folder `tests/e2e/` containing `test_e2e_pipeline.py`, `cases/` and `results/`. Marker: `pytestmark = pytest.mark.e2e`.
- Runs the whole pipeline Dopytywator → `CaseBrief` → Źródłowyszukiwacz → Pismogenerator through the same code the UI calls. Do not wire the modules together in the test. Nothing is mocked.
- One run uses one provider for every stage, unless the case says otherwise.
- `expected` covers every stage: the brief is valid and stays within `max_questions`; the expected sources are returned (or no relevant source); the letter is valid and has all required sections.
- A run passes only if every stage finished and every check passed. Producing some text is not a pass. The report also gives each stage's status and duration and saves the generated letter.

## Config

Add to `pyproject.toml` if missing, and add `tests/golden_set/*/results/` and `tests/e2e/results/` to `.gitignore`:

```toml
[tool.pytest.ini_options]
markers = [
  "golden: real-model tests of one module, run with `pytest -m golden`",
  "e2e: whole-pipeline tests, run with `pytest -m e2e`",
]
addopts = "--strict-markers -m 'not golden and not e2e'"
```