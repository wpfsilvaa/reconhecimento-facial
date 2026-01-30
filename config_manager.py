"""
Gerenciador de Configurações - Salva e carrega preferências do usuário
"""

import json
import os
from pathlib import Path
from datetime import datetime

CONFIG_DIR = "config"
CONFIG_FILE = os.path.join(CONFIG_DIR, "app_config.json")

# Configurações padrão
DEFAULT_CONFIG = {
    "camera": {
        "supported_resolutions": [(640, 480), (320, 240)],
        "current_resolution": "640x480",
        "current_fps": 30,
        "last_updated": None
    },
    "liveness": {
        "enabled": True,
        "threshold": 0.7
    },
    "recognition": {
        "interval": 0.15,
        "max_distance": None,
        "movement_detection": True
    },
    "performance": {
        "frame_size": "320x240"
    }
}


def ensure_config_dir():
    """Garante que o diretório de configuração existe."""
    try:
        Path(CONFIG_DIR).mkdir(exist_ok=True)
    except Exception as e:
        print(f"Erro ao criar diretório de config: {e}")


def load_config():
    """
    Carrega configurações do arquivo.
    Se o arquivo não existir, retorna configurações padrão.
    
    Returns:
        dict: Configurações carregadas
    """
    ensure_config_dir()
    
    if not os.path.exists(CONFIG_FILE):
        print(f"📝 Arquivo de config não encontrado. Usando padrões.")
        return DEFAULT_CONFIG.copy()
    
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"✅ Configurações carregadas de {CONFIG_FILE}")
        
        # Converter listas de volta para tuplas (JSON não suporta tuplas)
        if "camera" in config and "supported_resolutions" in config["camera"]:
            config["camera"]["supported_resolutions"] = [
                tuple(res) for res in config["camera"]["supported_resolutions"]
            ]
        
        return config
    except Exception as e:
        print(f"❌ Erro ao carregar config: {e}. Usando padrões.")
        return DEFAULT_CONFIG.copy()


def save_config(config):
    """
    Salva configurações no arquivo.
    
    Args:
        config (dict): Dicionário de configurações
    """
    ensure_config_dir()
    
    try:
        # Converter tuplas para listas (JSON não suporta tuplas)
        config_to_save = json.loads(json.dumps(config), object_hook=lambda x: x)
        
        # Garantir que resoluções sejam listas
        if "camera" in config_to_save and "supported_resolutions" in config_to_save["camera"]:
            config_to_save["camera"]["supported_resolutions"] = [
                list(res) if isinstance(res, tuple) else res 
                for res in config_to_save["camera"]["supported_resolutions"]
            ]
        
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config_to_save, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Configurações salvas em {CONFIG_FILE}")
    except Exception as e:
        print(f"❌ Erro ao salvar config: {e}")


def update_camera_config(supported_resolutions, current_resolution, current_fps):
    """
    Atualiza configurações de câmera e salva.
    
    Args:
        supported_resolutions: Lista de tuplas (width, height)
        current_resolution: String "WIDTHxHEIGHT"
        current_fps: int
    """
    config = load_config()
    
    config["camera"]["supported_resolutions"] = supported_resolutions
    config["camera"]["current_resolution"] = current_resolution
    config["camera"]["current_fps"] = current_fps
    config["camera"]["last_updated"] = datetime.now().isoformat()
    
    save_config(config)


def get_saved_resolutions():
    """
    Retorna resoluções salvas anteriormente.
    
    Returns:
        list: Lista de tuplas (width, height) ou None
    """
    config = load_config()
    resolutions = config.get("camera", {}).get("supported_resolutions", None)
    
    if resolutions and len(resolutions) > 0:
        print(f"📦 Usando {len(resolutions)} resoluções salvas")
        return resolutions
    
    return None


def get_saved_resolution_string():
    """Retorna a resolução salva como string."""
    config = load_config()
    return config.get("camera", {}).get("current_resolution", "640x480")


def get_saved_fps():
    """Retorna o FPS salvo."""
    config = load_config()
    return config.get("camera", {}).get("current_fps", 30)


def update_liveness_config(enabled, threshold):
    """Atualiza configurações de liveness."""
    config = load_config()
    config["liveness"]["enabled"] = enabled
    config["liveness"]["threshold"] = threshold
    save_config(config)


def update_recognition_config(interval, max_distance, movement_detection):
    """Atualiza configurações de reconhecimento."""
    config = load_config()
    config["recognition"]["interval"] = interval
    config["recognition"]["max_distance"] = max_distance
    config["recognition"]["movement_detection"] = movement_detection
    save_config(config)


def update_performance_config(frame_size):
    """Atualiza configurações de performance."""
    config = load_config()
    config["performance"]["frame_size"] = frame_size
    save_config(config)


def reset_config():
    """Reseta configurações para padrão."""
    try:
        if os.path.exists(CONFIG_FILE):
            os.remove(CONFIG_FILE)
        print("✅ Configurações resetadas")
    except Exception as e:
        print(f"❌ Erro ao resetar config: {e}")
