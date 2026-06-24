"""Calibrate FLOOR/CEILING thresholds against real embeddings from the project's memory bank.

Run via:
    uv run python scripts/calibrate_freshness_threshold.py
"""

from statistics import fmean

from openai import OpenAI

from slopometry.core.settings import settings
from slopometry.solo.services.memory_service import MemoryService


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def main() -> None:
    import os

    project_dir = os.environ.get("PROJECT_DIR", "/mnt/terradump/code/slopometry")
    service = MemoryService()
    memories = service.get_memories(project_dir=project_dir, limit=10000)
    print(f"Loaded {len(memories)} existing memories")

    if not memories:
        print("No memories to calibrate against. Run `solo find-memories --force` first.")
        return

    endpoint = settings.memory_embedding_endpoint
    model = settings.memory_embedding_model
    api_key = settings.memory_embedding_api_key.get_secret_value()
    if endpoint == "https://your-embedding-endpoint.com/v1":
        print("Embedding endpoint not configured.")
        return

    client = OpenAI(base_url=endpoint, api_key=api_key)
    embeddings: list[tuple[str, list[float]]] = []
    for m in memories:
        if m.embedding:
            embeddings.append((m.content, m.embedding))
            continue
        try:
            resp = client.embeddings.create(model=model, input=m.content)
            embeddings.append((m.content, resp.data[0].embedding))
        except Exception as e:
            print(f"  embed failed for {m.id}: {e}")

    if not embeddings:
        print("No embeddings available.")
        return

    print(f"Got {len(embeddings)} embeddings (dim={len(embeddings[0][1])})")
    sims: list[float] = []
    for i, (_, a) in enumerate(embeddings):
        for _, b in embeddings[i + 1 :]:
            sims.append(cosine(a, b))
    sims.sort()
    n = len(sims)

    def q(p: float) -> float:
        return sims[max(0, min(n - 1, int(n * p)))]

    print(f"n_pairs={n}")
    print(f"mean={fmean(sims):.4f}")
    print(f"p10={q(0.10):.4f}  p25={q(0.25):.4f}  p50={q(0.50):.4f}  p75={q(0.75):.4f}  p90={q(0.90):.4f}  p95={q(0.95):.4f}  p99={q(0.99):.4f}")
    print(f"max={sims[-1]:.4f}  min={sims[0]:.4f}")
    print()
    print("If p75 < FLOOR_THRESHOLD (0.45), threshold clamps to FLOOR. If p75 > CEILING_THRESHOLD (0.95), threshold clamps to CEILING.")
    print(f"Current derived_threshold would be p75={q(0.75):.4f} clamped to [0.45, 0.95] -> {max(0.45, min(0.95, q(0.75))):.4f}")


if __name__ == "__main__":
    main()
