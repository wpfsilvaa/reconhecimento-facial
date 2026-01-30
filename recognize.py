
import os
import numpy as np
from face_embedding import get_embedding

THRESHOLD = 10.0


def load_database():
    """
    Carrega todos os embeddings em uma única matriz (N x D),
    com um vetor de labels correspondente. Isso permite
    reconhecimento muito mais rápido com operações vetorizadas.
    """
    base_dir = "database/users"
    embeddings = []
    labels = []

    if not os.path.exists(base_dir):
        return {"embs": np.empty((0, 0)), "labels": np.array([])}

    for user in os.listdir(base_dir):
        user_dir = os.path.join(base_dir, user)
        if not os.path.isdir(user_dir):
            continue

        for f in os.listdir(user_dir):
            if not f.endswith(".npy"):
                continue
            emb = np.load(os.path.join(user_dir, f))
            embeddings.append(emb)
            labels.append(user)

    if not embeddings:
        return {"embs": np.empty((0, 0)), "labels": np.array([])}

    embs_matrix = np.vstack(embeddings)
    return {"embs": embs_matrix, "labels": np.array(labels)}


def recognize(frame, db):
    """
    Reconhece o usuário usando busca vetorizada:
    calcula a distância do embedding do frame para
    todos os embeddings do banco de uma vez.
    """
    emb = get_embedding(frame)

    embs = db.get("embs")
    labels = db.get("labels")

    if embs is None or embs.size == 0:
        return None, None

    # Distâncias Euclidianas em batch
    dists = np.linalg.norm(embs - emb, axis=1)
    best_idx = int(np.argmin(dists))
    best_dist = float(dists[best_idx])
    best_user = labels[best_idx]

    if best_dist < THRESHOLD:
        return best_user, best_dist
    return None, None
