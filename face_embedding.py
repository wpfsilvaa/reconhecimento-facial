import numpy as np
from deepface import DeepFace
from functools import lru_cache
import hashlib

# ===== Cache global do modelo =====
_model = None

def _get_model():
    global _model
    if _model is None:
        _model = DeepFace.build_model("Facenet")
    return _model


def _frame_hash(frame: np.ndarray) -> str:
    """
    Hash determinístico do conteúdo do frame.
    """
    return hashlib.sha256(frame.tobytes()).hexdigest()


@lru_cache(maxsize=128)
def _compute_embedding_cached(frame_hash: str, frame_bytes: bytes, shape: tuple):
    """
    Cache real de embeddings.
    Agora o reshape usa o shape correto do frame.
    """
    frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(shape)

    _get_model()

    reps = DeepFace.represent(
        img_path=frame,
        model_name="Facenet",
        detector_backend="opencv",
        enforce_detection=True,
    )

    return np.array(reps[0]["embedding"])


def get_embedding(frame: np.ndarray) -> np.ndarray:
    """
    API pública.
    """
    frame_hash = _frame_hash(frame)

    return _compute_embedding_cached(
        frame_hash,
        frame.tobytes(),
        frame.shape
    )
