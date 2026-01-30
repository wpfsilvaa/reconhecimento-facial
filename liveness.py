"""
Liveness Detection - Detecção de Vida para Segurança
Implementa múltiplas técnicas para validar que é uma pessoa real
"""

import cv2
import numpy as np
import time
from typing import Tuple, List, Dict

# Carrega os modelos de detecção
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)


class LivenessDetector:
    """
    Detector de liveness (vida) usando múltiplas técnicas
    """
    
    def __init__(self, confidence_threshold=0.7):
        """
        Args:
            confidence_threshold: Score mínimo (0-1) para considerar vivo
        """
        self.confidence_threshold = confidence_threshold
        self.blink_history = []
        self.head_position_history = []
        self.last_faces = []
        
    def detect_eye_blink(self, frame, face) -> Tuple[bool, float]:
        """
        Detecta piscar de olhos na face.
        Retorna: (blink_detected, confidence)
        
        Técnica: Calcula a relação entre altura e largura dos olhos
        Eyes Aspect Ratio (EAR) - valores baixos indicam olhos fechados
        """
        x, y, w, h = face
        face_roi = frame[y:y+h, x:x+w]
        
        # Detecta olhos na face
        eyes = eye_cascade.detectMultiScale(face_roi, 1.1, 4)
        
        if len(eyes) < 2:
            # Não conseguiu detectar 2 olhos
            return False, 0.3
        
        # Calcula Eye Aspect Ratio (EAR) para os 2 primeiros olhos
        ear_values = []
        for (ex, ey, ew, eh) in eyes[:2]:
            # EAR = (distância vertical) / (distância horizontal)
            eye_region = face_roi[ey:ey+eh, ex:ex+ew]
            
            # Detecta mudanças na intensidade dos pixels
            # Olhos abertos têm mais contraste
            _, thresh = cv2.threshold(cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY), 100, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                area = max([cv2.contourArea(c) for c in contours])
                ear = area / (ew * eh)
                ear_values.append(ear)
        
        if not ear_values:
            return False, 0.2
        
        avg_ear = np.mean(ear_values)
        
        # Limiar: EAR baixo = olho fechado
        # Se foi detectada uma variação, indica piscar
        blink_detected = False
        if len(self.blink_history) > 0:
            prev_ear = self.blink_history[-1]
            # Piscar = mudança significativa em EAR
            if abs(avg_ear - prev_ear) > 0.15:
                blink_detected = True
        
        self.blink_history.append(avg_ear)
        
        # Manter histórico dos últimos 10 frames
        if len(self.blink_history) > 10:
            self.blink_history.pop(0)
        
        confidence = min(avg_ear, 1.0)
        return blink_detected, confidence
    
    def detect_head_movement(self, frame, face) -> Tuple[bool, float]:
        """
        Detecta movimento natural de cabeça.
        Retorna: (movement_detected, confidence)
        
        Técnica: Rastreia posição do rosto entre frames
        """
        x, y, w, h = face
        center = (x + w//2, y + h//2)
        
        if len(self.head_position_history) > 0:
            prev_center = self.head_position_history[-1]
            
            # Calcula deslocamento
            displacement = np.sqrt(
                (center[0] - prev_center[0])**2 + 
                (center[1] - prev_center[1])**2
            )
            
            # Detectou movimento? (mais que 10 pixels)
            movement_detected = displacement > 10
        else:
            movement_detected = False
        
        self.head_position_history.append(center)
        
        # Manter histórico dos últimos 30 frames
        if len(self.head_position_history) > 30:
            self.head_position_history.pop(0)
        
        # Se tem histórico suficiente, analisa variabilidade
        if len(self.head_position_history) >= 5:
            positions = np.array(self.head_position_history)
            variance = np.var(positions, axis=0).mean()
            confidence = min(variance / 100, 1.0)  # Normaliza
        else:
            confidence = 0.5 if movement_detected else 0.2
        
        return movement_detected, confidence
    
    def detect_face_quality(self, frame, face) -> Tuple[bool, float]:
        """
        Valida qualidade da face detectada.
        Retorna: (quality_ok, confidence)
        
        Valida:
        - Tamanho da face (não muito pequena)
        - Luminosidade adequada
        - Não desfocada
        - Posição (não virada demais)
        """
        x, y, w, h = face
        face_roi = frame[y:y+h, x:x+w]
        
        # 1. Tamanho mínimo
        min_size = 100
        if w < min_size or h < min_size:
            return False, 0.2  # Face muito pequena
        
        # 2. Luminosidade
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        # Luminosidade ideal: 50-200
        if brightness < 50 or brightness > 200:
            return False, 0.4  # Muito escuro ou muito claro
        
        brightness_conf = 1.0 - abs(brightness - 125) / 125
        
        # 3. Detecta desfoque (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Variance alta = imagem clara (não desfocada)
        focus_conf = min(laplacian_var / 100, 1.0)
        
        if laplacian_var < 20:
            return False, 0.3  # Muito desfocado
        
        # 4. Posição da face (não muito virada)
        # Detecta se a face está frontal usando olhos
        eyes = eye_cascade.detectMultiScale(face_roi, 1.1, 4)
        
        if len(eyes) < 1:
            return False, 0.4  # Não conseguiu detectar olhos = face muito virada
        
        # Calcula scores
        quality_ok = (
            brightness > 50 and brightness < 200 and
            laplacian_var > 20 and
            len(eyes) >= 1
        )
        
        confidence = np.mean([brightness_conf, focus_conf])
        
        return quality_ok, confidence
    
    def validate_liveness(self, frames_list: List, face_list: List) -> Dict:
        """
        Valida liveness usando múltiplos frames.
        
        Args:
            frames_list: Lista de frames sequenciais
            face_list: Lista de faces detectadas em cada frame
        
        Returns:
            Dict com:
            - is_live: bool - É uma pessoa viva?
            - overall_confidence: float - Confiança geral (0-1)
            - blink_confidence: float - Confiança de piscar
            - movement_confidence: float - Confiança de movimento
            - quality_confidence: float - Confiança de qualidade
            - details: str - Detalhes da validação
        """
        
        if not frames_list or not face_list:
            return {
                "is_live": False,
                "overall_confidence": 0.0,
                "blink_confidence": 0.0,
                "movement_confidence": 0.0,
                "quality_confidence": 0.0,
                "details": "Sem frames ou faces"
            }
        
        blink_scores = []
        movement_scores = []
        quality_scores = []
        
        # Processa cada frame
        for frame, face in zip(frames_list, face_list):
            if face is not None:
                # Detecta piscar
                blink_detected, blink_conf = self.detect_eye_blink(frame, face)
                blink_scores.append(blink_conf)
                
                # Detecta movimento
                movement_detected, movement_conf = self.detect_head_movement(frame, face)
                movement_scores.append(movement_conf)
                
                # Valida qualidade
                quality_ok, quality_conf = self.detect_face_quality(frame, face)
                quality_scores.append(quality_conf)
        
        if not blink_scores:
            return {
                "is_live": False,
                "overall_confidence": 0.0,
                "blink_confidence": 0.0,
                "movement_confidence": 0.0,
                "quality_confidence": 0.0,
                "details": "Nenhuma face detectada nos frames"
            }
        
        # Calcula médias
        blink_conf = np.mean(blink_scores)
        movement_conf = np.mean(movement_scores)
        quality_conf = np.mean(quality_scores)
        
        # Score geral (média ponderada)
        overall = (
            blink_conf * 0.3 +      # 30% - piscar de olhos
            movement_conf * 0.3 +   # 30% - movimento de cabeça
            quality_conf * 0.4      # 40% - qualidade da face
        )
        
        is_live = overall >= self.confidence_threshold
        
        # Gera detalhes
        details = f"Piscar: {blink_conf:.1%} | Movimento: {movement_conf:.1%} | Qualidade: {quality_conf:.1%}"
        
        return {
            "is_live": is_live,
            "overall_confidence": overall,
            "blink_confidence": blink_conf,
            "movement_confidence": movement_conf,
            "quality_confidence": quality_conf,
            "details": details
        }
    
    def validate_frame(self, frame, face) -> Dict:
        """
        Valida um único frame para liveness.
        
        Uso: Durante reconhecimento em tempo real
        """
        
        if face is None:
            return {
                "is_live": False,
                "quality_ok": False,
                "confidence": 0.0
            }
        
        quality_ok, quality_conf = self.detect_face_quality(frame, face)
        
        # Tenta detectar piscar
        blink_detected, blink_conf = self.detect_eye_blink(frame, face)
        
        # Detecção de movimento
        movement_detected, movement_conf = self.detect_head_movement(frame, face)
        
        # Score geral (mais simples para frame único)
        confidence = (quality_conf * 0.4 + blink_conf * 0.3 + movement_conf * 0.3)
        
        is_live = confidence >= self.confidence_threshold and quality_ok
        
        return {
            "is_live": is_live,
            "confidence": confidence,
            "quality_ok": quality_ok,
            "blink_detected": blink_detected,
            "movement_detected": movement_detected
        }


# Função simplificada para uso fácil
def quick_liveness_check(frame, face, threshold=0.7) -> bool:
    """
    Verificação rápida de liveness em um frame.
    
    Args:
        frame: Frame OpenCV
        face: Tupla (x, y, w, h) da face
        threshold: Score mínimo (0-1)
    
    Returns:
        bool - É uma pessoa viva?
    """
    detector = LivenessDetector(confidence_threshold=threshold)
    result = detector.validate_frame(frame, face)
    return result["is_live"]


def challenge_based_liveness(cap, timeout=30) -> Tuple[bool, str]:
    """
    Validação de liveness baseada em desafio.
    Pede ao usuário que faça ações específicas:
    1. Piscar (5 segundos)
    2. Virar cabeça para esquerda (5 segundos)
    3. Virar cabeça para direita (5 segundos)
    
    Args:
        cap: VideoCapture
        timeout: Tempo máximo em segundos
    
    Returns:
        (success: bool, message: str)
    """
    
    detector = LivenessDetector(confidence_threshold=0.6)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    
    challenges = [
        ("Pisque os olhos rapidamente", 5),
        ("Vire a cabeça para a esquerda", 5),
        ("Vire a cabeça para a direita", 5),
    ]
    
    for challenge_text, duration in challenges:
        print(f"\n[Desafio Liveness] {challenge_text} ({duration}s)")
        
        frames = []
        faces = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.resize(frame, (320, 240))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(detected_faces) > 0:
                frames.append(frame)
                faces.append(detected_faces[0])
        
        if len(frames) < 3:
            return False, f"Falha na detecção de face durante desafio: {challenge_text}"
    
    # Se chegou aqui, realizou todos os desafios
    return True, "Validação de liveness bem-sucedida!"
