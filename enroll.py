import cv2, os
import numpy as np
import time
from face_embedding import get_embedding

# ===== OTIMIZAÇÃO #3: Remover detecção Haar dupla =====
# A detecção de rosto já é feita no get_embedding via DeepFace
# Removemos a redundância aqui mantendo apenas para feedback visual

# Função de cadastro guiado (versão terminal)
def guided_enroll(cap, user_id, face_cascade, instructions=None):
    """
    Cadastro de usuário com 6 poses diferentes.
    
    Otimizações aplicadas:
    - Detecção de rosto apenas visual (feedback ao usuário)
    - get_embedding já realiza detecção interna via DeepFace
    - Frames redimensionados para 320x240 para mais velocidade
    """
    if instructions is None:
        instructions = [
            "Olhe para frente",
            "Vire a cabeça para a direita",
            "Vire a cabeça para a esquerda",
            "Olhe para cima",
            "Olhe para baixo",
            "Sorria ou expressão neutra"
        ]

    user_dir = f"database/users/{user_id}"
    os.makedirs(user_dir, exist_ok=True)
    frames = []

    for i, instr in enumerate(instructions):
        print(f"{i+1}/{len(instructions)}: {instr}")
        input("Pressione Enter quando estiver pronto...")
        ret, frame = cap.read()
        if not ret:
            print("Falha na captura, tente novamente")
            continue
        # ===== OTIMIZAÇÃO #4: Redimensionar frame aqui também =====
        frame_resized = cv2.resize(frame, (320, 240))
        frames.append(frame_resized.copy())
        cv2.imshow("Cadastro", frame_resized)
        cv2.waitKey(500)

    cv2.destroyAllWindows()
    print("Processando embeddings...")

    # Gera embeddings em batch
    embeds_saved = 0
    for i, frame in enumerate(frames):
        # ===== OTIMIZAÇÃO #3: Removemos detecção Haar aqui =====
        # A detecção é feita internamente em get_embedding/DeepFace
        # Apenas informamos ao usuário
        try:
            emb = get_embedding(frame)
            np.save(f"{user_dir}/{i}.npy", emb)
            embeds_saved += 1
            print(f"  Frame {i}: embedding salvo ✓")
        except Exception as e:
            print(f"  Frame {i}: erro no embedding - {str(e)}")
            continue

    print(f"Cadastro concluído! {embeds_saved}/{len(frames)} embeddings salvos.")


# # Função de cadastro guiado (versão GUI)
# def guided_enroll_gui(cap, user_id, face_cascade, instructions, 
#                       progress_callback=None, status_callback=None, 
#                       frame_callback=None):
#     """
#     Versão do cadastro adaptada para GUI.
    
#     Args:
#         cap: Objeto VideoCapture
#         user_id: Nome do usuário
#         face_cascade: Detector de faces
#         instructions: Lista de instruções
#         progress_callback: Função para atualizar progresso (0.0 a 1.0)
#         status_callback: Função para atualizar status (texto, cor)
#         frame_callback: Função para atualizar frame de vídeo
#     """
#     if instructions is None:
#         instructions = [
#             "Olhe para frente",
#             "Vire a cabeça para a direita",
#             "Vire a cabeça para a esquerda",
#             "Olhe para cima",
#             "Olhe para baixo",
#             "Sorria ou expressão neutra"
#         ]

#     user_dir = f"database/users/{user_id}"
#     os.makedirs(user_dir, exist_ok=True)
#     frames = []
#     total_steps = len(instructions) * 2  # Captura + processamento

#     # Fase 1: Captura de frames
#     for i, instr in enumerate(instructions):
#         if status_callback:
#             status_callback(f"Passo {i+1}/{len(instructions)}: {instr}", "yellow")
        
#         if progress_callback:
#             progress_callback(i / total_steps)

#         # Aguardar 2 segundos para posicionamento
#         start_time = time.time()
#         while time.time() - start_time < 2.0:
#             ret, frame = cap.read()
#             if ret and frame_callback:
#                 frame_callback(frame)
#             time.sleep(0.05)

#         # Capturar frame final
#         ret, frame = cap.read()
#         if not ret:
#             if status_callback:
#                 status_callback(f"Falha na captura do passo {i+1}", "red")
#             continue

#         # Verificar se há rosto detectado
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#         if len(faces) == 0:
#             if status_callback:
#                 status_callback(f"Rosto não detectado no passo {i+1}. Tente novamente.", "red")
#             continue

#         frames.append(frame.copy())
#         if frame_callback:
#             frame_callback(frame)

#     if len(frames) == 0:
#         if status_callback:
#             status_callback("Nenhum frame válido capturado!", "red")
#         return

#     # Fase 2: Processamento de embeddings
#     if status_callback:
#         status_callback("Processando embeddings...", "blue")
    
#     saved_count = 0
#     new_embs = []
#     for i, frame in enumerate(frames):
#         if progress_callback:
#             progress_callback((len(instructions) + i) / total_steps)

#         # Aqui NÃO rechecamos o rosto, pois já foi validado na captura.
#         # Isso evita perder fotos válidas por variações do detector.
#         try:
#             emb = get_embedding(frame)
#             new_embs.append(emb)
#             np.save(f"{user_dir}/{saved_count}.npy", emb)
#             saved_count += 1
#         except Exception as e:
#             print(f"Erro ao processar frame {i}: {e}")
#             continue

#     if progress_callback:
#         progress_callback(1.0)

#     if status_callback:
#         status_callback(f"Cadastro concluído! {saved_count} embeddings salvos.", "green")

#     # Retorna embeddings novos para possível atualização incremental em memória
#     return np.array(new_embs) if new_embs else np.empty((0, 0))


# Função de cadastro guiado com captura manual
def guided_enroll_gui_manual(cap, user_id, face_cascade, instructions,
                              capture_event, progress_callback=None, 
                              status_callback=None, frame_getter=None,
                              thumbnail_callback=None):
    """
    Versão do cadastro com captura manual via botão.
    
    Args:
        cap: Objeto VideoCapture
        user_id: Nome do usuário
        face_cascade: Detector de faces
        instructions: Lista de instruções
        capture_event: threading.Event que é disparado quando o botão é clicado
        progress_callback: Função para atualizar progresso (0.0 a 1.0)
        status_callback: Função para atualizar status (texto, cor)
        frame_getter: Função que retorna o frame atual da câmera
        thumbnail_callback: Função para salvar primeira foto como miniatura
    """
    if instructions is None:
        instructions = [
            "Olhe para frente",
            "Vire a cabeça para a direita",
            "Vire a cabeça para a esquerda",
            "Olhe para cima",
            "Olhe para baixo",
            "Sorria ou expressão neutra"
        ]

    user_dir = f"database/users/{user_id}"
    os.makedirs(user_dir, exist_ok=True)
    frames = []
    total_steps = len(instructions) * 2  # Captura + processamento

    # Fase 1: Captura manual de frames
    for i, instr in enumerate(instructions):
        if status_callback:
            status_callback(f"Passo {i+1}/{len(instructions)}: {instr}\nClique em 'Capturar Foto' quando estiver pronto", "yellow")
        
        if progress_callback:
            progress_callback(i / total_steps)

        # Aguardar evento de captura (botão clicado)
        capture_event.clear()  # Limpar evento anterior
        capture_event.wait()  # Esperar até o botão ser clicado
        
        # Capturar frame quando o evento for disparado
        if frame_getter:
            frame = frame_getter()
        else:
            ret, frame = cap.read()
            if not ret:
                if status_callback:
                    status_callback(f"Falha na captura do passo {i+1}", "red")
                continue

        if frame is None:
            if status_callback:
                status_callback(f"Frame inválido no passo {i+1}", "red")
            continue

        # Verificar se há rosto detectado
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            if status_callback:
                status_callback(f"Rosto não detectado no passo {i+1}.\nTente novamente e clique em 'Capturar Foto'", "red")
            continue

        frames.append(frame.copy())
        if status_callback:
            status_callback(f"Foto {i+1} capturada! ✓", "green")
            time.sleep(0.5)  # Pequena pausa para feedback visual
        
        # Salvar primeira foto como miniatura
        if i == 0 and thumbnail_callback:
            thumbnail_callback(user_id, frame.copy())

    if len(frames) == 0:
        if status_callback:
            status_callback("Nenhum frame válido capturado!", "red")
        return

    # Fase 2: Processamento de embeddings
    if status_callback:
        status_callback("Processando embeddings...", "blue")
    
    saved_count = 0
    new_embs = []
    for i, frame in enumerate(frames):
        if progress_callback:
            progress_callback((len(instructions) + i) / total_steps)

        # Aqui também não rechecamos o rosto – já foi validado na captura.
        try:
            emb = get_embedding(frame)
            new_embs.append(emb)
            np.save(f"{user_dir}/{saved_count}.npy", emb)
            saved_count += 1
        except Exception as e:
            print(f"Erro ao processar frame {i}: {e}")
            continue

    if progress_callback:
        progress_callback(1.0)

    if status_callback:
        status_callback(f"Cadastro concluído! {saved_count} embeddings salvos.", "green")

    # Retorna embeddings novos para possível atualização incremental em memória
    return np.array(new_embs) if new_embs else np.empty((0, 0))