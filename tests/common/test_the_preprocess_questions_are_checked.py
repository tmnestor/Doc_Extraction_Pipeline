"""The question the preprocess task puts to the model is checked when it is read.

Until 26 November 2026 it was a key of the settings file, refused with its four parts when it was missing
or empty. It then moved to ``prompts/preprocess.yaml`` and lost both checks: an empty question was sent
with every photograph, and a file without one ended in a traceback.

Every question written into a file here is invented.
"""

from pathlib import Path

import pytest

from common.errors import ConfigError
from common.prompt_catalog import PromptCatalog

PARTS = ("What:", "Where:", "Expected:", "How to fix:")


def test_the_shipped_question_is_read() -> None:
    question = PromptCatalog().preprocess_prompt()

    assert question and question == question.strip()


@pytest.mark.parametrize(
    "written",
    [
        "prompt: ''\n",
        "prompt: '   '\n",
        "question: where is each receipt\n",
        "prompt: [where, is, each]\n",
        "- a list\n- and no mapping\n",
        "prompt: [unclosed\n",
        None,
    ],
)
def test_a_question_that_is_missing_empty_or_cannot_be_read_is_refused(
    tmp_path: Path, written: str | None
) -> None:
    if written is not None:
        (tmp_path / "preprocess.yaml").write_text(written)

    with pytest.raises(ConfigError) as refused:
        PromptCatalog(tmp_path).preprocess_prompt()

    message = str(refused.value)
    assert all(part in message for part in PARTS)
    assert str(tmp_path / "preprocess.yaml") in message and "prompt:" in message
