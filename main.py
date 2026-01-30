import cv2, threading, time
import numpy as np
from enroll import guided_enroll
from recognize import recognize, load_database
from camera import open_camera

# Detector de rosto Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ===== OTIMIZAÇÕES APLICADAS =====
# #2: Reduzir intervalo + detecção de movimento
# #3: Detectar rosto 1x, passar faces para recognize
# #4: Redimensionar frames para 320x240

def _frame_similarity(prev_frame, curr_frame, threshold=0.98):
    """
    Calcula similaridade entre dois frames usando diferença média.
    Se < threshold, frames são significativamente diferentes.
    
    Otimização #2: Pula reconhecimento se frame é muito similar ao anterior.
    """
    if prev_frame is None or curr_frame is None:
        return 0.0
    
    # Redimensiona para comparação rápida
    h, w = min(prev_frame.shape[0], curr_frame.shape[0]), min(prev_frame.shape[1], curr_frame.shape[1])
    p = cv2.resize(prev_frame[:h, :w], (64, 64))
    c = cv2.resize(curr_frame[:h, :w], (64, 64))
    
    diff = np.sum(np.abs(p.astype(float) - c.astype(float)))
    max_diff = 64 * 64 * 3 * 255
    similarity = 1.0 - (diff / max_diff)
    return similarity


# --- MAIN ---
mode = int(input("[1] Cadastro  [2] Reconhecimento: "))
cap = open_camera()

if mode == 1:
    user = input("Nome do usuário: ")
    guided_enroll(cap, user, face_cascade)

elif mode == 2:
    db = load_database()
    print("Reconhecimento iniciado (Otimizações: Cache Modelo, FPS++, 1x Detecção, Frame Redim)")

    # Variáveis compartilhadas
    frame_lock = threading.Lock()
    shared_frame = None
    shared_result = "Nao reconhecido"
    prev_frame = None
    recognition_count = 0
    skipped_count = 0

    # Thread de captura
    def capture_thread():
        global shared_frame
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            # ===== OTIMIZAÇÃO #4: Redimensionar frame para 320x240 =====
            frame_resized = cv2.resize(frame, (320, 240))
            with frame_lock:
                shared_frame = frame_resized.copy()

    # Thread de reconhecimento
    def recognition_thread():
        global shared_frame, shared_result, prev_frame, recognition_count, skipped_count
        # ===== OTIMIZAÇÃO #2: Intervalo de 150ms em vez de 500ms =====
        interval = 0.15  # 150ms = ~6.7 FPS, ainda otimizado
        last_time = 0
        
        while True:
            now = time.time()
            if now - last_time < interval:
                time.sleep(0.01)
                continue
            last_time = now

            with frame_lock:
                if shared_frame is None:
                    continue
                frame = shared_frame.copy()

            # ===== OTIMIZAÇÃO #2: Detecção de movimento para pular frames =====
            similarity = _frame_similarity(prev_frame, frame)
            if similarity > 0.98:
                skipped_count += 1
                prev_frame = frame.copy()
                continue  # Pula reconhecimento se frames são muito similares
            
            prev_frame = frame.copy()

            # ===== OTIMIZAÇÃO #3: Detectar rosto uma única vez aqui =====
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                shared_result = "Nao reconhecido"
                continue

            # Passa frame diretamente para recognize (já está redimensionado)
            user, dist = recognize(frame, db)
            recognition_count += 1
            
            if user:
                shared_result = f"{user} ({dist:.2f})"
            else:
                shared_result = "Nao reconhecido"

    # Start threads
    t1 = threading.Thread(target=capture_thread, daemon=True)
    t2 = threading.Thread(target=recognition_thread, daemon=True)
    t1.start()
    t2.start()

    # Interface principal
    fps_counter = 0
    fps_time = time.time()
    fps_display = 0
    
    while True:
        with frame_lock:
            if shared_frame is None:
                continue
            display_frame = shared_frame.copy()
            text = shared_result

        fps_counter += 1
        now = time.time()
        if now - fps_time >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_time = now

        cv2.putText(display_frame, text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if text != "Nao reconhecido" else (0, 0, 255), 2)
        cv2.putText(display_frame, f"FPS: {fps_display} | Recs: {recognition_count} | Skipped: {skipped_count}", 
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow("Reconhecimento", display_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()

