"""
Exemplo de Integração de Liveness Detection com o Sistema Existente
Mostra como adicionar segurança ao reconhecimento facial
"""

from liveness import LivenessDetector, challenge_based_liveness, quick_liveness_check
import cv2
from recognize import recognize, load_database
from camera import open_camera


def recognize_with_liveness(frame, face, db, detector=None, threshold=0.7):
    """
    Reconhecimento facial COM validação de liveness.
    
    Args:
        frame: Frame OpenCV
        face: Tupla (x, y, w, h) da face
        db: Database de embeddings
        detector: Instância de LivenessDetector (opcional)
        threshold: Score mínimo de confiança (0-1)
    
    Returns:
        {
            'user': str or None,
            'distance': float or None,
            'is_live': bool,
            'liveness_score': float,
            'allowed': bool  # True se reconhecido E vivo
        }
    """
    
    if detector is None:
        detector = LivenessDetector(confidence_threshold=threshold)
    
    # 1. Valida liveness primeiro
    liveness_result = detector.validate_frame(frame, face)
    is_live = liveness_result["is_live"]
    liveness_score = liveness_result["confidence"]
    
    # 2. Se não é vivo, nega acesso (segurança em primeiro lugar)
    if not is_live:
        return {
            'user': None,
            'distance': None,
            'is_live': False,
            'liveness_score': liveness_score,
            'allowed': False,
            'reason': f"Liveness validation falhou (score: {liveness_score:.1%})"
        }
    
    # 3. Se passou em liveness, faz o reconhecimento
    try:
        user, distance = recognize(frame, db)
    except Exception as e:
        return {
            'user': None,
            'distance': None,
            'is_live': True,
            'liveness_score': liveness_score,
            'allowed': False,
            'reason': f"Erro no reconhecimento: {str(e)}"
        }
    
    # 4. Reconhecimento bem-sucedido?
    allowed = (user is not None and is_live)
    
    return {
        'user': user,
        'distance': distance,
        'is_live': is_live,
        'liveness_score': liveness_score,
        'allowed': allowed,
        'reason': 'OK' if allowed else f'Reconhecimento falhou'
    }


def interactive_enrollment_with_liveness(cap, user_id, face_cascade, detector=None):
    """
    Cadastro interativo com validação de liveness.
    Garante que está cadastrando uma pessoa real (não foto).
    
    Args:
        cap: VideoCapture
        user_id: Nome do usuário
        face_cascade: Haar Cascade para detecção
        detector: LivenessDetector (opcional)
    
    Returns:
        bool - Cadastro bem-sucedido?
    """
    
    if detector is None:
        detector = LivenessDetector(confidence_threshold=0.6)
    
    print(f"\n[Cadastro com Liveness] Usuário: {user_id}")
    print("Instrução: Vire sua cabeça lentamente para provar que é uma pessoa real\n")
    
    # Coleta frames durante movimento de cabeça
    frames = []
    faces = []
    start_time = cv2.getTickCount()
    timeout = 15  # 15 segundos
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar frame")
            break
        
        frame = cv2.resize(frame, (320, 240))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(detected_faces) > 0:
            frames.append(frame)
            faces.append(detected_faces[0])
        
        # Mostra frame com status
        elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        cv2.putText(frame, f"Tempo: {elapsed:.1f}s", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Cadastro com Liveness", frame)
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESC para sair
            break
        
        if elapsed > timeout:
            break
    
    cv2.destroyAllWindows()
    
    # Valida liveness com os frames coletados
    if len(frames) < 5:
        print("❌ Falha: Não coletou frames suficientes")
        return False
    
    liveness_result = detector.validate_liveness(frames, faces)
    
    print(f"\n[Resultado Liveness]")
    print(f"  Confiança Geral: {liveness_result['overall_confidence']:.1%}")
    print(f"  {liveness_result['details']}")
    
    if not liveness_result['is_live']:
        print(f"❌ Falha: Não passou na validação de liveness")
        print(f"   Score: {liveness_result['overall_confidence']:.1%} < limiar")
        return False
    
    print(f"✅ Passou na validação de liveness!")
    print(f"   Você pode prosseguir com o cadastro")
    
    return True


def recognition_session_with_liveness(cap, db, detector=None, duration=60):
    """
    Sessão de reconhecimento com liveness detection ativo.
    
    Args:
        cap: VideoCapture
        db: Database de embeddings
        detector: LivenessDetector
        duration: Duração da sessão em segundos
    
    Yields:
        (user, confidence, is_live, allowed)
    """
    
    if detector is None:
        detector = LivenessDetector(confidence_threshold=0.7)
    
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    
    start_time = cv2.getTickCount()
    last_recognition_time = 0
    interval = 0.15  # 150ms entre reconhecimentos
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (320, 240))
        elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        
        if elapsed > duration:
            break
        
        # Intervalo de reconhecimento
        if elapsed - last_recognition_time < interval:
            continue
        
        last_recognition_time = elapsed
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            yield None, 0.0, False, False
            continue
        
        face = faces[0]
        
        # Reconhece COM liveness
        result = recognize_with_liveness(frame, face, db, detector)
        
        yield result['user'], result.get('distance', 0.0), result['is_live'], result['allowed']


# ============= EXEMPLO DE USO =============

if __name__ == "__main__":
    
    # Exemplo 1: Reconhecimento com Liveness
    print("=" * 60)
    print("EXEMPLO 1: Reconhecimento com Liveness Detection")
    print("=" * 60)
    
    cap = open_camera()
    db = load_database()
    detector = LivenessDetector(confidence_threshold=0.7)
    
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    
    print("\nCapturando 30 frames para teste...")
    for i in range(30):
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv2.resize(frame, (320, 240))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            result = recognize_with_liveness(frame, faces[0], db, detector)
            
            status = "✅ ACESSO PERMITIDO" if result['allowed'] else "❌ ACESSO NEGADO"
            print(f"Frame {i}: {status}")
            print(f"  Usuario: {result['user']}")
            print(f"  Vivo: {result['is_live']} (score: {result['liveness_score']:.1%})")
            print(f"  Motivo: {result.get('reason', 'OK')}")
            print()
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print("TESTE CONCLUÍDO")
    print("=" * 60)
