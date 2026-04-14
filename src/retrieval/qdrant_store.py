from __future__ import annotations

import atexit
import uuid
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

from src.models import ModalityRecord, SearchHit


_POINT_ID_NAMESPACE = uuid.UUID("3f4e14a9-527f-4e33-9f3d-e5c8e4ce5d5f")


def _stable_point_id(record: ModalityRecord) -> str:
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, record.record_id))


class QdrantStore:
    def __init__(self, *, path: str, collection_prefix: str, embedding_dim: int) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(self.path))
        self.collection_prefix = collection_prefix
        self.embedding_dim = embedding_dim
        atexit.register(self.close)

    def collection_name(self, modality: str) -> str:
        return f"{self.collection_prefix}_{modality}"

    def ensure_collection(self, modality: str) -> None:
        name = self.collection_name(modality)
        if self.client.collection_exists(name):
            return
        self.client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=self.embedding_dim,
                distance=models.Distance.COSINE,
            ),
        )

    def recreate_collection(self, modality: str) -> None:
        name = self.collection_name(modality)
        if self.client.collection_exists(name):
            self.client.delete_collection(name)
        self.ensure_collection(modality)

    def upsert_records(
        self,
        modality: str,
        records: list[ModalityRecord],
        embeddings: np.ndarray,
        batch_size: int = 128,
    ) -> None:
        self.ensure_collection(modality)
        name = self.collection_name(modality)
        points: list[models.PointStruct] = []
        for record, embedding in zip(records, embeddings):
            payload = {
                "record_id": record.record_id,
                "video_file": record.video_file,
                "modality": record.modality,
                "start": record.start,
                "end": record.end,
                "text": record.text,
                "metadata": record.metadata,
            }
            if record.metadata.get("det_type"):
                payload["det_type"] = record.metadata["det_type"]
            points.append(
                models.PointStruct(
                    id=_stable_point_id(record),
                    vector=embedding.tolist(),
                    payload=payload,
                )
            )

        for start in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=name,
                points=points[start : start + batch_size],
            )

    def search(
        self,
        modality: str,
        query_vector: np.ndarray,
        top_k: int,
        filter_payload: dict[str, object] | None = None,
    ) -> list[SearchHit]:
        query_filter = None
        if filter_payload:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value),
                    )
                    for key, value in filter_payload.items()
                ]
            )
        hits = self.client.query_points(
            collection_name=self.collection_name(modality),
            query=query_vector.tolist(),
            limit=top_k,
            query_filter=query_filter,
        ).points
        return [
            SearchHit(
                video_file=point.payload["video_file"],
                modality=point.payload["modality"],
                start=float(point.payload["start"]),
                end=float(point.payload["end"]),
                score=float(point.score),
                text=point.payload["text"],
                metadata=point.payload.get("metadata") or {},
            )
            for point in hits
        ]

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass
