"""Public parser abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from models import Document


class DocumentParser(ABC):
    """Translate one local original into the unified :class:`Document` model."""

    @abstractmethod
    def parse(self, source: Path, *, document_id: str, source_url: str | None = None) -> Document:
        """Parse ``source`` without modifying its bytes."""

