"""
Gerenciador de Usuários - Lista, deleta e adiciona fotos a usuários
"""

import os
import shutil
from pathlib import Path

USERS_DIR = "database/users"


def get_all_users():
    """
    Retorna lista de todos os usuários cadastrados.
    
    Returns:
        list: Lista de tuplas (username, num_embeddings)
    """
    if not os.path.exists(USERS_DIR):
        return []
    
    users = []
    for username in os.listdir(USERS_DIR):
        user_path = os.path.join(USERS_DIR, username)
        if not os.path.isdir(user_path):
            continue
        
        # Contar embeddings (.npy files)
        embeddings = [f for f in os.listdir(user_path) if f.endswith('.npy')]
        num_embeddings = len(embeddings)
        
        users.append((username, num_embeddings))
    
    # Ordenar por nome
    users.sort(key=lambda x: x[0])
    return users


def delete_user(username):
    """
    Deleta um usuário e todos seus embeddings.
    
    Args:
        username (str): Nome do usuário a deletar
    
    Returns:
        tuple: (success: bool, message: str)
    """
    user_path = os.path.join(USERS_DIR, username)
    
    if not os.path.exists(user_path):
        return False, f"Usuário '{username}' não encontrado"
    
    try:
        shutil.rmtree(user_path)
        return True, f"✅ Usuário '{username}' deletado com sucesso"
    except Exception as e:
        return False, f"❌ Erro ao deletar '{username}': {str(e)}"


def delete_embedding(username, embedding_index):
    """
    Deleta um embedding específico de um usuário.
    
    Args:
        username (str): Nome do usuário
        embedding_index (int): Índice do embedding a deletar (0, 1, 2, ...)
    
    Returns:
        tuple: (success: bool, message: str)
    """
    user_path = os.path.join(USERS_DIR, username)
    embedding_file = os.path.join(user_path, f"{embedding_index}.npy")
    
    if not os.path.exists(embedding_file):
        return False, f"Embedding {embedding_index} não encontrado para '{username}'"
    
    try:
        os.remove(embedding_file)
        return True, f"✅ Embedding {embedding_index} deletado"
    except Exception as e:
        return False, f"❌ Erro ao deletar embedding: {str(e)}"


def get_user_embeddings(username):
    """
    Retorna lista de índices de embeddings de um usuário.
    
    Args:
        username (str): Nome do usuário
    
    Returns:
        list: Lista de índices [0, 1, 2, ...]
    """
    user_path = os.path.join(USERS_DIR, username)
    
    if not os.path.exists(user_path):
        return []
    
    embeddings = []
    for f in os.listdir(user_path):
        if f.endswith('.npy'):
            try:
                idx = int(f.replace('.npy', ''))
                embeddings.append(idx)
            except ValueError:
                continue
    
    return sorted(embeddings)


def user_exists(username):
    """Verifica se um usuário existe."""
    user_path = os.path.join(USERS_DIR, username)
    return os.path.isdir(user_path)


def get_user_info(username):
    """
    Retorna informações detalhadas sobre um usuário.
    
    Returns:
        dict: {'username': str, 'num_embeddings': int, 'embeddings': []}
    """
    if not user_exists(username):
        return None
    
    embeddings = get_user_embeddings(username)
    
    return {
        'username': username,
        'num_embeddings': len(embeddings),
        'embeddings': embeddings
    }
