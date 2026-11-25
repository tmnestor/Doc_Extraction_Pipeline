"""A job that says what type every page is gives its pages that type, and classify then asks no model.

Some sets of pages are all of one type, known beforehand: the receipts of a public dataset, say. Until
25 November 2026 the one such set the pipeline read had a task of its own, with its own prompt, parser
and scorer. Now a job reads such a set by naming the type (``document_type=<a document type>``):
``classify`` writes each page a record of that type without starting an engine, and the stages after it
read the pages as they read any others.

The pages are blank images made by the bench, and every reply is invented.
"""

from pathlib import Path

import pytest

from bench import AN_ENGINE_ON_EACH_CARD, ONE_ENGINE, Bench
from bench import run_settings
from common.stage_timing import read_stage_timings
from stages import classify
from stages.io import read_jsonl


def given(bench: Bench, output: Path, kind: str, model: str = ONE_ENGINE) -> Path:
    return classify.run(
        output, classify.checked(bench.pages, run_settings(model, bench.models)), given_type=kind
    )


@pytest.mark.parametrize("model", [ONE_ENGINE, AN_ENGINE_ON_EACH_CARD])
def test_a_dataset_gives_every_page_its_type_and_no_model_is_asked(
    bench: Bench, tmp_path: Path, model: str, caplog: pytest.LogCaptureFixture
) -> None:
    output = tmp_path / "out" / "classifications.jsonl"

    with caplog.at_level("INFO"):
        given(bench, output, "RECEIPT", model)

    records = read_jsonl(output)
    assert [r["image_name"] for r in records] == bench.names
    assert {(r["document_type"], r["type_from"], r["raw_response"]) for r in records} == {
        ("RECEIPT", "dataset", "")
    }
    assert all(r["processing_time"] == 0.0 and "error" not in r for r in records)
    assert bench.engines_started == 0 and bench.processor.attempts == []
    assert read_stage_timings(output.parent) == []
    assert "6 page(s) given the type RECEIPT by the dataset: the model is not asked" in caplog.text


def test_the_record_is_the_one_every_stage_after_reads(bench: Bench, tmp_path: Path) -> None:
    """The same keys in the same order as a record of a page the model was asked about."""
    asked = read_jsonl(bench.classify(tmp_path / "asked" / "classifications.jsonl"))[0]

    told = read_jsonl(given(bench, tmp_path / "told" / "classifications.jsonl", "RECEIPT"))[0]

    assert list(told) == list(asked)


def test_pages_the_model_had_typed_keep_their_records(bench: Bench, tmp_path: Path) -> None:
    """The stage carries on: only the pages its output does not hold are given the type."""
    output = tmp_path / "out" / "classifications.jsonl"
    bench.classify(output)
    kept = output.read_text().splitlines()[:4]
    output.write_text("".join(line + "\n" for line in kept))

    given(bench, output, "RECEIPT")

    assert output.read_text().splitlines()[:4] == kept
    assert [r["type_from"] for r in read_jsonl(output)[4:]] == ["dataset", "dataset"]
    assert [r["image_name"] for r in read_jsonl(output)] == bench.names


def test_given_a_second_time_nothing_is_written(bench: Bench, tmp_path: Path) -> None:
    output = given(bench, tmp_path / "out" / "classifications.jsonl", "RECEIPT")
    before = output.stat().st_mtime_ns

    given(bench, output, "RECEIPT")

    assert output.stat().st_mtime_ns == before


def test_the_stages_after_read_the_dataset_s_pages_as_their_type(bench: Bench, tmp_path: Path) -> None:
    out = tmp_path / "out"
    classified = given(bench, out / "classifications.jsonl", "RECEIPT")

    raw = bench.extract(classified, out / "raw_extractions.jsonl")
    cleaned = bench.clean(raw, out / "cleaned_extractions.jsonl")
    results = read_jsonl(
        bench.evaluate(cleaned, bench.answer_key(out / "ground_truth.jsonl"), out / "evaluation")
    )

    assert {r["prompt_used"] for r in read_jsonl(raw)} == {"receipt"}
    assert bench.processor.calls == [[name for name in bench.names[i : i + 2]] for i in (0, 2, 4)]
    assert len(results) == len(bench.names) and all("median_f1" in r for r in results)


def test_a_type_the_schema_file_does_not_have_stops_classify_before_anything_is_written(
    bench: Bench, tmp_path: Path
) -> None:
    from common.errors import ConfigError
    from common.stage_settings import resolve

    job = {
        "model": ONE_ENGINE,
        "image_dir": str(bench.pages),
        "output": str(tmp_path / "out"),
        "DOC_EXTRACT_MODELS_DIR": str(bench.models),
        "document_type": "LEDGER",
    }

    with pytest.raises(ConfigError) as refused:
        classify.check(resolve("classify", job))

    message = str(refused.value)
    assert all(part in message for part in ("What:", "Where:", "Expected:", "How to fix:"))
    assert "LEDGER" in message and "document_type" in message and "sroie_receipt" in message
    assert not (tmp_path / "out").exists() and bench.engines_started == 0


@pytest.mark.parametrize("written", ["SROIE_RECEIPT", "sroie_receipt"])
def test_a_type_of_the_schema_file_in_either_case_passes_the_check(
    bench: Bench, tmp_path: Path, written: str
) -> None:
    from common.stage_settings import resolve

    job = {
        "model": ONE_ENGINE,
        "image_dir": str(bench.pages),
        "output": str(tmp_path / "out"),
        "DOC_EXTRACT_MODELS_DIR": str(bench.models),
        "document_type": written,
    }

    assert len(classify.check(resolve("classify", job)).images) == len(bench.names)
