"""Value objects for bank statement extraction pipeline.

Pure dataclasses with zero project dependencies — safe to import anywhere.
Extracted from unified_bank_extractor.py to enable independent testing
of correction and filtering logic without VLM/torch/PIL dependencies.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class ExtractionStrategy(Enum):
    """Available extraction strategies."""

    BALANCE_DESCRIPTION = auto()  # 2-turn: header detection + balance extraction
    AMOUNT_DESCRIPTION = auto()  # 2-turn: header + amount extraction (negative = debit)
    DEBIT_CREDIT_DESCRIPTION = auto()  # 2-turn: header + debit/credit extraction (no balance)
    TABLE_EXTRACTION = auto()  # 3-turn: header + format classify + table


# How many model calls a bank statement gets: pipeline.processing.bank_strategy.
BANK_STRATEGY_TWO_TURN = "two_turn"  # header detection, then a layout-specific extraction
BANK_STRATEGY_SINGLE_PASS = "single_pass"  # one call with the schema prompt, reply kept as written
BANK_STRATEGIES = (BANK_STRATEGY_TWO_TURN, BANK_STRATEGY_SINGLE_PASS)


def validate_bank_strategy(value: Any, config_path: Any = None) -> str:
    """Return ``value`` if it names a bank strategy, else fail with a diagnostic.

    Args:
        value: The configured ``pipeline.processing.bank_strategy``.
        config_path: The run config it was read from, named in the error. None
            when the caller holds only the merged config.

    Returns:
        The validated strategy name.

    Raises:
        ValueError: If the value is missing or not one of ``BANK_STRATEGIES``.
    """
    if isinstance(value, str) and value in BANK_STRATEGIES:
        return value
    where = config_path if config_path is not None else "the run config (config/run_config.yml)"
    raise ValueError(
        f"What: bank_strategy is {value!r}, which is not a bank extraction strategy.\n"
        f"Where: {where} -> pipeline.processing.bank_strategy\n"
        f"Expected: one of {', '.join(BANK_STRATEGIES)}, e.g.:\n"
        f"  pipeline:\n"
        f"    processing:\n"
        f"      bank_strategy: {BANK_STRATEGY_TWO_TURN}\n"
        f"How to fix: set bank_strategy under 'pipeline.processing:' in that file to "
        f"'{BANK_STRATEGY_TWO_TURN}' (header detection, then extraction) or "
        f"'{BANK_STRATEGY_SINGLE_PASS}' (one call per statement)."
    )


@dataclass
class ColumnMapping:
    """Mapped column names from detected headers."""

    date: str | None = None
    description: str | None = None
    debit: str | None = None
    credit: str | None = None
    balance: str | None = None
    amount: str | None = None

    @property
    def has_balance(self) -> bool:
        """Check if balance column was detected."""
        return self.balance is not None

    def to_dict(self) -> dict[str, str | None]:
        """Serialize to dict for JSON compatibility."""
        return {
            "date": self.date,
            "description": self.description,
            "debit": self.debit,
            "credit": self.credit,
            "balance": self.balance,
            "amount": self.amount,
        }


@dataclass
class ExtractionResult:
    """Standardized extraction result from UnifiedBankExtractor."""

    document_type: str = "BANK_STATEMENT"
    statement_date_range: str = "NOT_FOUND"
    transaction_dates: list[str] = field(default_factory=list)
    line_item_descriptions: list[str] = field(default_factory=list)
    transaction_amounts_paid: list[str] = field(default_factory=list)
    account_balances: list[str] = field(default_factory=list)

    # Metadata
    strategy_used: str = ""
    turns_executed: int = 0
    headers_detected: list[str] = field(default_factory=list)
    column_mapping: ColumnMapping | None = None
    raw_responses: dict[str, str] = field(default_factory=dict)
    correction_stats: Any = None

    def to_schema_dict(self) -> dict[str, str]:
        """Convert to schema format with pipe-delimited fields."""
        return {
            "DOCUMENT_TYPE": self.document_type,
            "STATEMENT_DATE_RANGE": self.statement_date_range,
            "TRANSACTION_DATES": " | ".join(self.transaction_dates)
            if self.transaction_dates
            else "NOT_FOUND",
            "LINE_ITEM_DESCRIPTIONS": " | ".join(self.line_item_descriptions)
            if self.line_item_descriptions
            else "NOT_FOUND",
            "TRANSACTION_AMOUNTS_PAID": " | ".join(self.transaction_amounts_paid)
            if self.transaction_amounts_paid
            else "NOT_FOUND",
            "ACCOUNT_BALANCE": " | ".join(self.account_balances) if self.account_balances else "NOT_FOUND",
        }

    def to_metadata_dict(self) -> dict[str, Any]:
        """Build metadata dict for pipeline compatibility."""
        meta: dict[str, Any] = {
            "strategy_used": self.strategy_used,
            "turns_executed": self.turns_executed,
            "headers_detected": self.headers_detected,
            "column_mapping": self.column_mapping.to_dict() if self.column_mapping else {},
            "raw_responses": self.raw_responses,
        }
        if self.correction_stats:
            meta["correction_stats"] = str(self.correction_stats)
        return meta
