"""
Script de Benchmark - Plano de Melhoria de Performance
Mede latência de embedding, FPS e uso de recursos antes/depois otimizações.

Execução:
    python benchmark.py [modo]
    
Modo:
    1: Medir latência de embedding (100 iterações)
    2: Medir FPS com captura de câmera (30 segundos)
    3: Medir performance de reconhecimento (30 segundos)
"""

import cv2
import time
import numpy as np
import psutil
import os
from face_embedding import get_embedding
from recognize import load_database, recognize
from camera import open_camera

def measure_embedding_latency(n_iterations=100):
    """
    Mede latência de embedding em ms.
    
    Baseline: ~500-1000ms (sem cache)
    Meta:     ~100-200ms (com cache)
    """
    print(f"\n=== BENCHMARK: Latência de Embedding ({n_iterations} iterações) ===")
    
    cap = open_camera()
    ret, frame = cap.read()
    if not ret:
        print("Falha na captura de câmera")
        return
    
    # Redimensiona como no sistema
    frame = cv2.resize(frame, (320, 240))
    
    # Warmup: carrega modelo
    print("Aquecendo modelo...")
    try:
        _ = get_embedding(frame)
    except Exception as e:
        print(f"Erro no warmup: {e}")
        cap.release()
        return
    
    # Medição
    latencies = []
    print(f"Medindo {n_iterations} embeddings...")
    
    for i in range(n_iterations):
        if i % 10 == 0:
            print(f"  {i}/{n_iterations}...", end="\r")
        
        start = time.perf_counter()
        try:
            _ = get_embedding(frame)
        except Exception as e:
            print(f"Erro na iteração {i}: {e}")
            continue
        elapsed = (time.perf_counter() - start) * 1000  # Converter para ms
        latencies.append(elapsed)
    
    cap.release()
    
    if not latencies:
        print("Nenhuma medição bem-sucedida")
        return
    
    # Estatísticas
    latencies = np.array(latencies)
    print(f"\n✓ Resultados ({len(latencies)} sucessos):")
    print(f"  Mínimo:     {latencies.min():.1f}ms")
    print(f"  Máximo:     {latencies.max():.1f}ms")
    print(f"  Média:      {latencies.mean():.1f}ms")
    print(f"  Mediana:    {np.median(latencies):.1f}ms")
    print(f"  Desvio Pad: {latencies.std():.1f}ms")
    print(f"  P95:        {np.percentile(latencies, 95):.1f}ms")
    print(f"  P99:        {np.percentile(latencies, 99):.1f}ms")
    
    return latencies


def measure_fps(duration_seconds=30):
    """
    Mede FPS com captura de câmera.
    
    Baseline: ~2 FPS (500ms interval)
    Meta:     ~15-20 FPS (150ms interval + movimento)
    """
    print(f"\n=== BENCHMARK: FPS de Captura ({duration_seconds}s) ===")
    
    cap = open_camera()
    if not cap.isOpened():
        print("Falha ao abrir câmera")
        return
    
    frame_count = 0
    start_time = time.perf_counter()
    
    print(f"Capturando por {duration_seconds}s...")
    while True:
        now = time.perf_counter()
        if now - start_time >= duration_seconds:
            break
        
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (320, 240))  # Simula otimização
            frame_count += 1
        else:
            print("Falha na captura")
            break
    
    elapsed = time.perf_counter() - start_time
    cap.release()
    
    fps = frame_count / elapsed
    print(f"\n✓ Resultados:")
    print(f"  Frames capturados: {frame_count}")
    print(f"  Tempo total: {elapsed:.1f}s")
    print(f"  FPS: {fps:.1f} frames/segundo")
    print(f"  Intervalo médio: {(1.0/fps)*1000:.1f}ms")
    
    return fps


def measure_recognition_performance(duration_seconds=30):
    """
    Mede performance completa de reconhecimento.
    
    Inclui: captura + detecção + embedding + matching
    """
    print(f"\n=== BENCHMARK: Performance de Reconhecimento ({duration_seconds}s) ===")
    
    cap = open_camera()
    db = load_database()
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    
    if not cap.isOpened():
        print("Falha ao abrir câmera")
        return
    
    frame_count = 0
    recognition_count = 0
    total_latency = 0
    recognitions_latency = []
    
    print(f"Reconhecendo por {duration_seconds}s...")
    start_time = time.perf_counter()
    
    while True:
        now = time.perf_counter()
        if now - start_time >= duration_seconds:
            break
        
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv2.resize(frame, (320, 240))
        frame_count += 1
        
        # Detecta rosto
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            # Mede latência de reconhecimento
            start_rec = time.perf_counter()
            try:
                user, dist = recognize(frame, db)
                elapsed_rec = (time.perf_counter() - start_rec) * 1000
                recognitions_latency.append(elapsed_rec)
                recognition_count += 1
            except Exception as e:
                print(f"Erro no reconhecimento: {e}")
    
    elapsed = time.perf_counter() - start_time
    cap.release()
    
    print(f"\n✓ Resultados:")
    print(f"  Frames capturados: {frame_count}")
    print(f"  Reconhecimentos: {recognition_count}")
    print(f"  Tempo total: {elapsed:.1f}s")
    print(f"  FPS captura: {frame_count/elapsed:.1f}")
    print(f"  Taxa reconhecimento: {recognition_count/elapsed:.1f} rec/s")
    
    if recognitions_latency:
        rec_lat = np.array(recognitions_latency)
        print(f"\n  Latência de reconhecimento:")
        print(f"    Mínimo:  {rec_lat.min():.1f}ms")
        print(f"    Máximo:  {rec_lat.max():.1f}ms")
        print(f"    Média:   {rec_lat.mean():.1f}ms")
        print(f"    Mediana: {np.median(rec_lat):.1f}ms")
    
    return fps_val


def check_system_resources():
    """Mostra uso de recursos do sistema."""
    print(f"\n=== RECURSOS DO SISTEMA ===")
    
    # CPU
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"CPU: {cpu_percent}%")
    
    # Memória
    mem = psutil.virtual_memory()
    print(f"Memória: {mem.used/1024/1024:.0f}MB / {mem.total/1024/1024:.0f}MB ({mem.percent}%)")
    
    # Processo atual
    proc = psutil.Process(os.getpid())
    print(f"Processo (Face Recognition):")
    print(f"  CPU: {proc.cpu_percent():.1f}%")
    print(f"  Memória: {proc.memory_info().rss/1024/1024:.0f}MB")


def main():
    print("=" * 60)
    print("BENCHMARK - Face Recognition System")
    print("=" * 60)
    
    print("\nOPÇÕES:")
    print("  1: Latência de Embedding (100 iterações)")
    print("  2: FPS de Captura (30s)")
    print("  3: Performance de Reconhecimento (30s)")
    print("  4: Tudo + Recursos")
    
    mode = input("\nEscolha (1-4): ").strip()
    
    if mode == "1":
        measure_embedding_latency(100)
    elif mode == "2":
        measure_fps(30)
    elif mode == "3":
        measure_recognition_performance(30)
    elif mode == "4":
        measure_embedding_latency(100)
        print("\n" + "=" * 60)
        measure_fps(30)
        print("\n" + "=" * 60)
        measure_recognition_performance(30)
        print("\n" + "=" * 60)
        check_system_resources()
    else:
        print("Opção inválida")
    
    print("\n" + "=" * 60)
    print("BENCHMARK CONCLUÍDO")
    print("=" * 60)


if __name__ == "__main__":
    main()
