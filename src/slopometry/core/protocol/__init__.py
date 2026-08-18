"""Extracted hook protocol: harness-independent event ingestion.

- `kinds`: canonical `EventKind` taxonomy and open `KnownSource` constants.
- `schema`: `EventEnvelope`, the stable external contract for agent tools.
- `ingest`: expansion and idempotent storage of envelope batches.
- `adapters`: mappings from harness-specific event names to canonical kinds.

`ingest` is intentionally not re-exported here to avoid a circular import
with the database layer; import it from `slopometry.core.protocol.ingest`.
"""

from slopometry.core.protocol.kinds import EventKind, KnownSource
from slopometry.core.protocol.schema import EventEnvelope

__all__ = ["EventEnvelope", "EventKind", "KnownSource"]
