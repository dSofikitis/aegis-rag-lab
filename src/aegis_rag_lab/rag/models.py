from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentInput:
    source: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentChunk:
    id: str
    source: str
    content: str
    metadata: dict[str, Any]
    embedding: list[float] | None = None

    def citation(self) -> str:
        chunk_index = self.metadata.get("chunk_index")
        if chunk_index is None:
            return self.source
        return f"{self.source}#chunk-{chunk_index}"
