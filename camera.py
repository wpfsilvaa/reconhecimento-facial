
import cv2

def get_camera_supported_resolutions(cap, fps=30):
    """
    Descobre dinamicamente as resoluções suportadas pela câmera.
    Testa uma lista de resoluções conhecidas e retorna as que funcionam.
    
    Args:
        cap: VideoCapture object
        fps: FPS desejado (padrão 30)
    
    Returns:
        List de tuplas (width, height) suportadas, em ordem decrescente
    """
    # Lista de resoluções para testar (em ordem de preferência, maior primeiro)
    candidates = [
        (1920, 1080),  # Full HD
        (1280, 720),   # HD
        (960, 720),    # SVGA
        (800, 600),    # SVGA
        (640, 480),    # VGA
        (480, 360),    # nHD
        (320, 240),    # QVGA
        (160, 120),    # QQVGA
    ]
    
    supported = []
    
    print("🔍 Detectando resoluções suportadas pela câmera...")
    
    for width, height in candidates:
        # Tentar configurar
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        
        # Verificar o que foi realmente configurado
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Se conseguiu a resolução desejada, adiciona à lista
        if actual_width == width and actual_height == height:
            supported.append((width, height))
            print(f"  ✅ {width}x{height}")
        else:
            print(f"  ❌ {width}x{height} (câmera retornou {actual_width}x{actual_height})")
    
    if supported:
        print(f"📊 Resoluções suportadas: {len(supported)}")
        return supported
    else:
        print("⚠️  Nenhuma resolução padrão funcionou. Usando configuração padrão da câmera.")
        return []


def get_camera_supported_fps(cap, width=640, height=480):
    """
    Descobre os FPS suportados pela câmera em uma determinada resolução.
    
    Args:
        cap: VideoCapture object
        width: Largura (padrão 640)
        height: Altura (padrão 480)
    
    Returns:
        List de FPS suportados
    """
    # Primeiro configura a resolução
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # Lista de FPS para testar
    fps_candidates = [60, 48, 30, 24, 15, 10]
    
    supported_fps = []
    
    print(f"🔍 Detectando FPS suportados para {width}x{height}...")
    
    for fps in fps_candidates:
        cap.set(cv2.CAP_PROP_FPS, fps)
        actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        if actual_fps == fps:
            supported_fps.append(fps)
            print(f"  ✅ {fps} FPS")
        else:
            print(f"  ❌ {fps} FPS (câmera retornou {actual_fps} FPS)")
    
    if supported_fps:
        print(f"📊 FPS suportados: {supported_fps}")
        return supported_fps
    else:
        print("⚠️  Nenhum FPS padrão funcionou.")
        return []


def validate_camera_config(cap, width, height, fps):
    """
    Valida se a câmera conseguiu configurar corretamente.
    Se falhar, tenta resoluções alternativas.
    
    Returns: (success: bool, actual_width, actual_height, actual_fps)
    """
    # Configurar câmera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    # Validar se conseguiu configurar
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Se conseguiu configurar a resolução solicitada, retorna sucesso
    if actual_width == width and actual_height == height:
        return True, actual_width, actual_height, actual_fps
    
    # Se não conseguiu, usa as resoluções suportadas detectadas dinamicamente
    supported_resolutions = get_camera_supported_resolutions(cap, fps)
    
    if supported_resolutions:
        # Tenta a primeira resolução suportada (a maior)
        fallback_width, fallback_height = supported_resolutions[0]
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, fallback_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, fallback_height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"⚠️  Câmera não suporta {width}x{height}. Usando {actual_width}x{actual_height}.")
        return True, actual_width, actual_height, actual_fps
    
    print(f"❌ Falha ao configurar câmera")
    return False, actual_width, actual_height, actual_fps


def open_camera():
    cap = cv2.VideoCapture(0)
    success, w, h, fps = validate_camera_config(cap, 640, 480, 30)
    if not success:
        print(f"⚠️  Câmera aberta com resolução: {w}x{h} @ {fps}FPS")
    return cap
