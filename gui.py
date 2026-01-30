import cv2
import threading
import time
import os
from datetime import datetime, timedelta
import customtkinter as ctk
from PIL import Image, ImageTk
import numpy as np
from enroll import guided_enroll_gui_manual
from recognize import recognize, load_database
from face_embedding import get_embedding
from camera import open_camera

# ===== SEGURANÇA: Liveness Detection =====
from liveness import LivenessDetector
from liveness_integration import recognize_with_liveness

# ===== GERENCIADOR DE CONFIGURAÇÕES =====
from config_manager import (
    load_config, save_config, get_saved_resolutions,
    get_saved_resolution_string, get_saved_fps,
    update_camera_config, update_liveness_config,
    update_recognition_config, update_performance_config
)

# ===== GERENCIADOR DE USUÁRIOS =====
from user_manager import get_all_users, delete_user, get_user_info

# ===== OTIMIZAÇÕES APLICADAS =====
# #2: Reduzir intervalo + detecção de movimento
# #4: Redimensionar frames para 320x240

def _frame_similarity(prev_frame, curr_frame, threshold=0.98):
    """
    Calcula similaridade entre dois frames.
    Se > threshold, frames são muito similares.
    
    Otimização: Pula reconhecimento se frame é muito similar ao anterior.
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

# Configuração do CustomTkinter
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Detector de rosto Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Diretório e configuração de logs
LOG_DIR = "logs"


class FaceRecognitionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Sistema de Reconhecimento Facial")
        self.geometry("1000x700")
        self.resizable(True, True)

        # ===== CARREGAR CONFIGURAÇÕES SALVAS =====
        print("[INFO] Carregando configuracoes...")
        config = load_config()
        
        # Variáveis compartilhadas
        self.cap = None
        self.frame_lock = threading.Lock()
        self.shared_frame = None
        self.shared_result = "Nao reconhecido"
        self.is_recognition_running = False
        self.is_enrollment_running = False
        self.db = None
        
        # ===== OTIMIZAÇÃO #2: Variáveis para detecção de movimento =====
        self.prev_frame_recognition = None
        self.recognition_count = 0
        self.skipped_count = 0

        # ===== SEGURANÇA: Liveness Detection =====
        liveness_config = config.get("liveness", {})
        liveness_threshold = liveness_config.get("threshold", 0.7)
        self.liveness_detector = LivenessDetector(confidence_threshold=liveness_threshold)
        self.liveness_enabled = liveness_config.get("enabled", True)
        self.liveness_stats = {
            'checks': 0,
            'passed': 0,
            'failed': 0
        }

        # ===== CONFIGURAÇÕES AJUSTÁVEIS =====
        recognition_config = config.get("recognition", {})
        self.recognition_interval = recognition_config.get("interval", 0.15)
        self.max_distance = recognition_config.get("max_distance", None)
        self.movement_detection_enabled = recognition_config.get("movement_detection", True)
        
        perf_config = config.get("performance", {})
        frame_size_str = perf_config.get("frame_size", "320x240")
        size_map = {
            "160x120": (120, 160),
            "320x240": (240, 320),
            "640x480": (480, 640),
        }
        self.frame_size = size_map.get(frame_size_str, (240, 320))
        
        # ===== CONFIGURAÇÕES DE CÂMERA =====
        camera_config = config.get("camera", {})
        resolution_str = camera_config.get("current_resolution", "640x480")
        self.camera_width, self.camera_height = map(int, resolution_str.split('x'))
        self.camera_fps = camera_config.get("current_fps", 30)

        # Info de última pessoa reconhecida
        self.last_recognized_user = "Nenhum"
        self.last_logged_hour = None  # hora do último log
        self.last_logged_user = None  # usuário do último log

        # Variáveis para cadastro manual
        self.capture_event = threading.Event()
        self.current_enroll_frame = None
        self.enroll_frame_lock = threading.Lock()
        self.current_enrollment_instruction = ""  # Instrução atual do cadastro

        # Janela do cliente (segunda tela)
        self.client_window = None
        self.client_video_label = None
        self.client_status_label = None

        # UI de logs
        self.log_textbox = None

        # Preparar diretório de logs e limpeza de arquivos antigos
        self.ensure_logs_dir()
        self.cleanup_old_logs()

        # ===== INICIALIZAR CONFIGURAÇÕES DE CÂMERA NA PRIMEIRA EXECUÇÃO =====
        self.initialize_camera_config_on_startup()

        # Criar abas
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)

        # Aba de Cadastro
        self.enroll_tab = self.tabview.add("Cadastro")
        self.setup_enroll_tab()

        # Aba de Reconhecimento
        self.recognize_tab = self.tabview.add("Reconhecimento")
        self.setup_recognize_tab()

        # Aba de Usuários
        self.users_tab = self.tabview.add("Usuários")
        self.setup_users_tab()

        # Aba de Logs
        self.log_tab = self.tabview.add("Logs")
        self.setup_log_tab()

        # Aba de Configurações
        self.config_tab = self.tabview.add("Configurações")
        self.setup_config_tab()

        # Inicializar câmera
        self.init_camera()

        # Carregar logs do dia na aba de logs
        self.load_today_log_to_ui()

    def init_camera(self):
        """Inicializa a câmera"""
        try:
            self.cap = open_camera()
            if self.cap is None or not self.cap.isOpened():
                self.show_error("Erro ao abrir a câmera")
                return False
            return True
        except Exception as e:
            self.show_error(f"Erro ao inicializar câmera: {str(e)}")
            return False

    def initialize_camera_config_on_startup(self):
        """
        Gera configurações na primeira inicialização.
        Se não existir arquivo de config, detecta resoluções e salva.
        """
        # Verificar se arquivo de config existe
        if os.path.exists("config/app_config.json"):
            print("[OK] Configuracoes ja existem")
            return
        
        print("[INFO] Primeira inicializacao - Detectando resolucoes da camera...")
        
        # Executar detecção de forma SÍNCRONA (bloqueia até terminar)
        self._detect_and_save_config()

    def _detect_and_save_config(self):
        """Detecta e salva configurações na primeira inicialização."""
        try:
            from camera import get_camera_supported_resolutions
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                print("⚠️  Câmera não aberta. Usando configurações padrão.")
                # Salvar padrão
                update_camera_config([(640, 480), (320, 240)], "640x480", 30)
                return
            
            print("[INFO] Detectando resolucoes suportadas...")
            resolutions = get_camera_supported_resolutions(cap, 30)
            cap.release()
            
            if resolutions:
                # Salvar resoluções detectadas
                update_camera_config(resolutions, "640x480", 30)
                print(f"✅ Configurações salvas! ({len(resolutions)} resoluções detectadas)")
            else:
                # Se não detectou nada, usar fallback
                update_camera_config([(640, 480), (320, 240)], "640x480", 30)
                print("⚠️  Nenhuma resolução padrão funcionou. Usando fallback.")
        
        except Exception as e:
            print(f"❌ Erro ao detectar configurações: {e}")
            # Salvar padrão mesmo em caso de erro
            update_camera_config([(640, 480), (320, 240)], "640x480", 30)

    def _detect_supported_resolutions(self):
        """Detecta automaticamente as resoluções suportadas pela câmera."""
        # Tentar carregar resoluções salvas primeiro
        saved_resolutions = get_saved_resolutions()
        if saved_resolutions:
            print(f"♻️  Usando resoluções do cache ({len(saved_resolutions)} resoluções)")
            return saved_resolutions
        
        # Se não tiver cache, detectar
        try:
            from camera import get_camera_supported_resolutions
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("⚠️  Câmera não aberta. Usando fallback.")
                return [(640, 480), (320, 240)]
            
            resolutions = get_camera_supported_resolutions(cap, 30)
            cap.release()
            
            if resolutions:
                # Salvar resoluções detectadas
                update_camera_config(resolutions, "640x480", 30)
                return resolutions
            else:
                return [(640, 480), (320, 240)]
        except Exception as e:
            print(f"Erro ao detectar resoluções: {e}")
            return [(640, 480), (320, 240)]

    def setup_enroll_tab(self):
        """Configura a aba de cadastro"""
        # Frame principal
        main_frame = ctk.CTkFrame(self.enroll_tab)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Lado esquerdo - Controles
        left_frame = ctk.CTkFrame(main_frame)
        left_frame.pack(side="left", fill="both", padx=(0, 10), pady=10)

        # Título
        title_label = ctk.CTkLabel(
            left_frame, text="Cadastro de Usuário", font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(pady=(10, 20))

        # Campo de nome
        name_label = ctk.CTkLabel(left_frame, text="Nome do usuário:")
        name_label.pack(pady=(0, 5))
        self.name_entry = ctk.CTkEntry(left_frame, width=250, height=35)
        self.name_entry.pack(pady=(0, 20))

        # Instruções
        instructions_label = ctk.CTkLabel(
            left_frame, text="Instruções:", font=ctk.CTkFont(size=16, weight="bold")
        )
        instructions_label.pack(pady=(0, 10))

        self.instruction_text = ctk.CTkTextbox(
            left_frame, width=250, height=200, wrap="word"
        )
        self.instruction_text.pack(pady=(0, 20))
        self.instruction_text.insert("1.0", "1. Olhe para frente\n2. Vire a cabeça para a direita\n3. Vire a cabeça para a esquerda\n4. Olhe para cima\n5. Olhe para baixo\n6. Sorria ou expressão neutra")
        self.instruction_text.configure(state="disabled")

        # Botão de cadastro
        self.enroll_button = ctk.CTkButton(
            left_frame,
            text="Iniciar Cadastro",
            command=self.start_enrollment,
            width=250,
            height=40,
            font=ctk.CTkFont(size=16),
        )
        self.enroll_button.pack(pady=(0, 10))

        # Botão de capturar foto (inicialmente oculto)
        self.capture_button = ctk.CTkButton(
            left_frame,
            text="📷 Capturar Foto",
            command=self.capture_photo,
            width=250,
            height=50,
            font=ctk.CTkFont(size=18, weight="bold"),
            fg_color="green",
            hover_color="darkgreen",
        )
        self.capture_button.pack(pady=(0, 10))
        self.capture_button.pack_forget()  # Oculto inicialmente

        # Status do cadastro
        self.enroll_status = ctk.CTkLabel(
            left_frame, text="", font=ctk.CTkFont(size=14), text_color="gray"
        )
        self.enroll_status.pack()

        # Progresso
        self.enroll_progress = ctk.CTkProgressBar(left_frame, width=250)
        self.enroll_progress.pack(pady=(10, 0))
        self.enroll_progress.set(0)

        # Lado direito - Preview da câmera
        right_frame = ctk.CTkFrame(main_frame)
        right_frame.pack(side="right", fill="both", expand=True, pady=10)

        camera_label = ctk.CTkLabel(
            right_frame, text="Preview da Câmera", font=ctk.CTkFont(size=18)
        )
        camera_label.pack(pady=(10, 5))

        self.enroll_video_label = ctk.CTkLabel(right_frame, text="")
        self.enroll_video_label.pack(pady=10, padx=10)
        
        # Iniciar thread de preview da câmera na aba de cadastro
        self.enroll_preview_running = False
        self.start_enroll_preview()

    def setup_recognize_tab(self):
        """Configura a aba de reconhecimento"""
        # Frame principal
        main_frame = ctk.CTkFrame(self.recognize_tab)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Frame de controles
        control_frame = ctk.CTkFrame(main_frame)
        control_frame.pack(fill="x", padx=10, pady=10)

        # Botão de iniciar/parar reconhecimento
        self.recognize_button = ctk.CTkButton(
            control_frame,
            text="Iniciar Reconhecimento",
            command=self.toggle_recognition,
            width=180,
            height=40,
            font=ctk.CTkFont(size=16),
        )
        self.recognize_button.pack(side="left", padx=10, pady=10)

        # Botão de recarregar banco
        reload_button = ctk.CTkButton(
            control_frame,
            text="Recarregar Banco",
            command=self.reload_database,
            width=150,
            height=40,
            font=ctk.CTkFont(size=14),
        )
        reload_button.pack(side="left", padx=10, pady=10)

        # ===== SEGURANÇA: Toggle de Liveness =====
        self.liveness_toggle = ctk.CTkSwitch(
            control_frame,
            text="🔒 Validação de Liveness",
            command=self.toggle_liveness,
            font=ctk.CTkFont(size=12),
        )
        self.liveness_toggle.pack(side="left", padx=10, pady=10)
        self.liveness_toggle.select()  # Começa ativado

        # Slider para ajustar threshold de liveness em tempo real
        self.liveness_threshold_label = ctk.CTkLabel(
            control_frame,
            text=f"Threshold: {self.liveness_detector.confidence_threshold:.2f}",
            font=ctk.CTkFont(size=12),
            text_color="gray",
        )
        self.liveness_threshold_label.pack(side="left", padx=(8, 4))

        self.liveness_threshold_slider = ctk.CTkSlider(
            control_frame,
            from_=0.40,
            to=0.95,
            number_of_steps=55,
            command=self.set_liveness_threshold,
            width=140,
        )
        # Inicializa a posição do slider com o valor atual
        try:
            self.liveness_threshold_slider.set(self.liveness_detector.confidence_threshold)
        except Exception:
            self.liveness_threshold_slider.set(0.7)
        self.liveness_threshold_slider.pack(side="left", padx=4, pady=10)

        # Botão para abrir tela do cliente (segunda janela)
        client_button = ctk.CTkButton(
            control_frame,
            text="Abrir tela do cliente",
            command=self.open_client_window,
            width=170,
            height=40,
            font=ctk.CTkFont(size=14),
        )
        client_button.pack(side="left", padx=10, pady=10)

        # Status
        self.recognize_status = ctk.CTkLabel(
            control_frame,
            text="Pronto",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="gray",
        )
        self.recognize_status.pack(side="left", padx=20, pady=10)

        # Última pessoa reconhecida
        self.last_recognized_label = ctk.CTkLabel(
            control_frame,
            text="Última pessoa reconhecida: Nenhum",
            font=ctk.CTkFont(size=14),
            text_color="white",
        )
        self.last_recognized_label.pack(side="left", padx=20, pady=10)

    def setup_config_tab(self):
        """Configura a aba de configurações com ajustes do sistema."""
        # Frame principal com scroll
        main_frame = ctk.CTkFrame(self.config_tab)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Scrollable frame
        scrollable_frame = ctk.CTkScrollableFrame(main_frame, fg_color="transparent")
        scrollable_frame.pack(fill="both", expand=True)

        # Título
        title = ctk.CTkLabel(
            scrollable_frame,
            text="Configurações do Sistema",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        title.pack(pady=(0, 20), padx=10)

        # ===== SEÇÃO: DETECÇÃO DE LIVENESS =====
        liveness_label = ctk.CTkLabel(
            scrollable_frame,
            text="🛡️ Detecção de Liveness (Anti-spoofing)",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#FFD700",
        )
        liveness_label.pack(pady=(15, 10), padx=10, anchor="w")

        liveness_frame = ctk.CTkFrame(scrollable_frame, fg_color="#1a1a1a", corner_radius=8)
        liveness_frame.pack(fill="x", padx=10, pady=(0, 15))

        # Threshold
        threshold_container = ctk.CTkFrame(liveness_frame, fg_color="transparent")
        threshold_container.pack(fill="x", padx=15, pady=(10, 5))

        threshold_label = ctk.CTkLabel(
            threshold_container,
            text="Threshold de Confiança:",
            font=ctk.CTkFont(size=11),
        )
        threshold_label.pack(side="left", padx=(0, 10))

        self.threshold_display = ctk.CTkLabel(
            threshold_container,
            text=f"{self.liveness_detector.confidence_threshold:.2f}",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#00FF7F",
        )
        self.threshold_display.pack(side="right")

        self.config_threshold_slider = ctk.CTkSlider(
            liveness_frame,
            from_=0.40,
            to=0.95,
            number_of_steps=55,
            command=self._set_config_threshold,
        )
        self.config_threshold_slider.set(self.liveness_detector.confidence_threshold)
        self.config_threshold_slider.pack(fill="x", padx=15, pady=(0, 10))

        threshold_info = ctk.CTkLabel(
            liveness_frame,
            text="Quanto maior, mais rigoroso (bloqueia mais fotos). Recomendado: 0.70-0.75",
            font=ctk.CTkFont(size=9),
            text_color="gray",
        )
        threshold_info.pack(padx=15, pady=(0, 10), anchor="w")

        # Toggle Liveness
        toggle_frame = ctk.CTkFrame(liveness_frame, fg_color="transparent")
        toggle_frame.pack(fill="x", padx=15, pady=(5, 10))

        toggle_label = ctk.CTkLabel(
            toggle_frame,
            text="Ativar Detecção de Liveness:",
            font=ctk.CTkFont(size=11),
        )
        toggle_label.pack(side="left", padx=(0, 10))

        self.liveness_toggle = ctk.CTkSwitch(
            toggle_frame,
            text="",
            command=self._toggle_liveness_config,
            onvalue=True,
            offvalue=False,
        )
        self.liveness_toggle.pack(side="right")
        self.liveness_toggle.select() if self.liveness_enabled else self.liveness_toggle.deselect()

        # ===== SEÇÃO: RECONHECIMENTO =====
        recognize_label = ctk.CTkLabel(
            scrollable_frame,
            text="🎯 Configurações de Reconhecimento",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#FFD700",
        )
        recognize_label.pack(pady=(15, 10), padx=10, anchor="w")

        recognize_frame = ctk.CTkFrame(scrollable_frame, fg_color="#1a1a1a", corner_radius=8)
        recognize_frame.pack(fill="x", padx=10, pady=(0, 15))

        # Intervalo de reconhecimento
        interval_container = ctk.CTkFrame(recognize_frame, fg_color="transparent")
        interval_container.pack(fill="x", padx=15, pady=(10, 5))

        interval_label = ctk.CTkLabel(
            interval_container,
            text="Intervalo de Reconhecimento:",
            font=ctk.CTkFont(size=11),
        )
        interval_label.pack(side="left", padx=(0, 10))

        self.interval_display = ctk.CTkLabel(
            interval_container,
            text=f"{0.15:.3f}s",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#00FF7F",
        )
        self.interval_display.pack(side="right")

        self.interval_slider = ctk.CTkSlider(
            recognize_frame,
            from_=0.05,
            to=0.50,
            number_of_steps=45,
            command=self._set_recognition_interval,
        )
        self.interval_slider.set(0.15)
        self.interval_slider.pack(fill="x", padx=15, pady=(0, 10))

        interval_info = ctk.CTkLabel(
            recognize_frame,
            text="Tempo entre reconhecimentos. Menor = mais rápido mas mais processamento.",
            font=ctk.CTkFont(size=9),
            text_color="gray",
        )
        interval_info.pack(padx=15, pady=(0, 10), anchor="w")

        # Distância máxima
        distance_container = ctk.CTkFrame(recognize_frame, fg_color="transparent")
        distance_container.pack(fill="x", padx=15, pady=(10, 5))

        distance_label = ctk.CTkLabel(
            distance_container,
            text="Distância Máxima para Reconhecimento:",
            font=ctk.CTkFont(size=11),
        )
        distance_label.pack(side="left", padx=(0, 10))

        self.distance_display = ctk.CTkLabel(
            distance_container,
            text="∞ (ilimitado)",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#00FF7F",
        )
        self.distance_display.pack(side="right")

        self.distance_slider = ctk.CTkSlider(
            recognize_frame,
            from_=0.0,
            to=1.0,
            number_of_steps=100,
            command=self._set_max_distance,
        )
        self.distance_slider.set(0.0)
        self.distance_slider.pack(fill="x", padx=15, pady=(0, 10))

        distance_info = ctk.CTkLabel(
            recognize_frame,
            text="Quanto menor, mais restritivo (só reconhece rostos muito similares). 0 = sem limite.",
            font=ctk.CTkFont(size=9),
            text_color="gray",
        )
        distance_info.pack(padx=15, pady=(0, 10), anchor="w")

        # ===== SEÇÃO: PERFORMANCE =====
        perf_label = ctk.CTkLabel(
            scrollable_frame,
            text="⚡ Configurações de Performance",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#FFD700",
        )
        perf_label.pack(pady=(15, 10), padx=10, anchor="w")

        perf_frame = ctk.CTkFrame(scrollable_frame, fg_color="#1a1a1a", corner_radius=8)
        perf_frame.pack(fill="x", padx=10, pady=(0, 15))

        # Detecção de movimento
        movement_container = ctk.CTkFrame(perf_frame, fg_color="transparent")
        movement_container.pack(fill="x", padx=15, pady=(10, 5))

        movement_label = ctk.CTkLabel(
            movement_container,
            text="Detecção de Movimento (Otimização):",
            font=ctk.CTkFont(size=11),
        )
        movement_label.pack(side="left", padx=(0, 10))

        self.movement_toggle = ctk.CTkSwitch(
            movement_container,
            text="",
            command=self._toggle_movement_detection,
            onvalue=True,
            offvalue=False,
        )
        self.movement_toggle.pack(side="right")
        self.movement_toggle.select()

        movement_info = ctk.CTkLabel(
            perf_frame,
            text="Pula reconhecimento se câmera não detecta movimento. Reduz processamento.",
            font=ctk.CTkFont(size=9),
            text_color="gray",
        )
        movement_info.pack(padx=15, pady=(0, 10), anchor="w")

        # Redimensionamento de frame
        resize_container = ctk.CTkFrame(perf_frame, fg_color="transparent")
        resize_container.pack(fill="x", padx=15, pady=(10, 5))

        resize_label = ctk.CTkLabel(
            resize_container,
            text="Tamanho de Frame para Processamento:",
            font=ctk.CTkFont(size=11),
        )
        resize_label.pack(side="left", padx=(0, 10))

        self.resize_var = ctk.StringVar(value="320x240")
        self.resize_dropdown = ctk.CTkComboBox(
            resize_container,
            values=["160x120", "320x240", "640x480", "Full HD"],
            variable=self.resize_var,
            command=self._set_frame_size,
            width=120,
        )
        self.resize_dropdown.pack(side="right")

        resize_info = ctk.CTkLabel(
            perf_frame,
            text="Tamanho menor = mais rápido. 320x240 recomendado para equilíbrio.",
            font=ctk.CTkFont(size=9),
            text_color="gray",
        )
        resize_info.pack(padx=15, pady=(0, 10), anchor="w")

        # ===== SEÇÃO: CÂMERA =====
        camera_label = ctk.CTkLabel(
            scrollable_frame,
            text="📷 Configurações de Câmera",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#FFD700",
        )
        camera_label.pack(pady=(15, 10), padx=10, anchor="w")

        camera_frame = ctk.CTkFrame(scrollable_frame, fg_color="#1a1a1a", corner_radius=8)
        camera_frame.pack(fill="x", padx=10, pady=(0, 15))

        # Detectar resoluções suportadas
        self.supported_resolutions = self._detect_supported_resolutions()
        resolution_strings = [f"{w}x{h}" for w, h in self.supported_resolutions]
        
        # Resolução da câmera
        resolution_container = ctk.CTkFrame(camera_frame, fg_color="transparent")
        resolution_container.pack(fill="x", padx=15, pady=(10, 5))

        resolution_label = ctk.CTkLabel(
            resolution_container,
            text="Resolução:",
            font=ctk.CTkFont(size=11),
        )
        resolution_label.pack(side="left", padx=(0, 10))

        current_res = f"{self.camera_width}x{self.camera_height}"
        self.resolution_var = ctk.StringVar(value=current_res if current_res in resolution_strings else (resolution_strings[0] if resolution_strings else "640x480"))
        self.resolution_dropdown = ctk.CTkComboBox(
            resolution_container,
            values=resolution_strings if resolution_strings else ["640x480", "320x240"],
            variable=self.resolution_var,
            width=120,
        )
        self.resolution_dropdown.pack(side="right")

        resolution_info = ctk.CTkLabel(
            camera_frame,
            text="Resoluções detectadas automaticamente de sua câmera",
            font=ctk.CTkFont(size=9),
            text_color="gray",
        )
        resolution_info.pack(padx=15, pady=(0, 10), anchor="w")

        # FPS da câmera
        fps_container = ctk.CTkFrame(camera_frame, fg_color="transparent")
        fps_container.pack(fill="x", padx=15, pady=(10, 5))

        fps_label = ctk.CTkLabel(
            fps_container,
            text="FPS (Frames por Segundo):",
            font=ctk.CTkFont(size=11),
        )
        fps_label.pack(side="left", padx=(0, 10))

        self.fps_var = ctk.StringVar(value=str(self.camera_fps))
        self.fps_dropdown = ctk.CTkComboBox(
            fps_container,
            values=["15", "24", "30", "60"],
            variable=self.fps_var,
            width=100,
        )
        self.fps_dropdown.pack(side="right")

        camera_info = ctk.CTkLabel(
            camera_frame,
            text="Clique em 'Aplicar Configurações' para ativar as mudanças na câmera.",
            font=ctk.CTkFont(size=9),
            text_color="gray",
        )
        camera_info.pack(padx=15, pady=(5, 10), anchor="w")

        # Botão Aplicar
        apply_button_frame = ctk.CTkFrame(camera_frame, fg_color="transparent")
        apply_button_frame.pack(fill="x", padx=15, pady=(5, 10))

        self.apply_camera_btn = ctk.CTkButton(
            apply_button_frame,
            text="Aplicar Configurações de Câmera",
            command=self._apply_camera_settings,
            fg_color="#2d7a2d",
            hover_color="#3a9a3a",
            width=250,
        )
        self.apply_camera_btn.pack(side="left")

        # ===== SEÇÃO: INFORMAÇÕES E AÇÕES =====
        stats_label = ctk.CTkLabel(
            scrollable_frame,
            text="📊 Estatísticas e Ações",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#FFD700",
        )
        stats_label.pack(pady=(15, 10), padx=10, anchor="w")

        stats_frame = ctk.CTkFrame(scrollable_frame, fg_color="#1a1a1a", corner_radius=8)
        stats_frame.pack(fill="x", padx=10, pady=(0, 15))

        # Estatísticas
        stats_display_frame = ctk.CTkFrame(stats_frame, fg_color="transparent")
        stats_display_frame.pack(fill="x", padx=15, pady=(10, 0))

        self.config_checks_label = ctk.CTkLabel(
            stats_display_frame,
            text=f"Verificações de Liveness: {self.liveness_stats['checks']}",
            font=ctk.CTkFont(size=10),
        )
        self.config_checks_label.pack(side="left", padx=(0, 20))

        self.config_passed_label = ctk.CTkLabel(
            stats_display_frame,
            text=f"✅ Passou: {self.liveness_stats['passed']}",
            font=ctk.CTkFont(size=10),
            text_color="#00FF7F",
        )
        self.config_passed_label.pack(side="left", padx=(0, 20))

        self.config_failed_label = ctk.CTkLabel(
            stats_display_frame,
            text=f"❌ Bloqueado: {self.liveness_stats['failed']}",
            font=ctk.CTkFont(size=10),
            text_color="#FF6B6B",
        )
        self.config_failed_label.pack(side="left")

        # Botão para resetar
        button_frame = ctk.CTkFrame(stats_frame, fg_color="transparent")
        button_frame.pack(fill="x", padx=15, pady=(15, 10))

        reset_btn = ctk.CTkButton(
            button_frame,
            text="Resetar Estatísticas",
            command=self._reset_stats,
            fg_color="#4a4a4a",
            hover_color="#5a5a5a",
            width=200,
        )
        reset_btn.pack(side="left", padx=(0, 10))

        # ===== NOTAS =====
        notes_label = ctk.CTkLabel(
            scrollable_frame,
            text="ℹ️ Alterações são aplicadas imediatamente. Algumas configurações podem impactar a performance.",
            font=ctk.CTkFont(size=9),
            text_color="gray",
        )
        notes_label.pack(pady=(20, 10), padx=10, anchor="w")

    def _set_config_threshold(self, value):
        """Callback para slider de threshold na aba de configurações."""
        val = float(value)
        self.liveness_detector.confidence_threshold = val
        self.threshold_display.configure(text=f"{val:.2f}")
        # Salvar automaticamente
        update_liveness_config(self.liveness_enabled, val)

    def _set_recognition_interval(self, value):
        """Callback para slider de intervalo de reconhecimento."""
        val = float(value)
        self.recognition_interval = val
        self.interval_display.configure(text=f"{val:.3f}s")
        # Salvar automaticamente
        update_recognition_config(val, self.max_distance, self.movement_detection_enabled)

    def _set_max_distance(self, value):
        """Callback para slider de distância máxima."""
        val = float(value)
        if val == 0.0:
            self.max_distance = None
            self.distance_display.configure(text="∞ (ilimitado)")
        else:
            self.max_distance = val
            self.distance_display.configure(text=f"{val:.2f}")
        # Salvar automaticamente
        update_recognition_config(self.recognition_interval, self.max_distance, self.movement_detection_enabled)

    def _toggle_liveness_config(self):
        """Callback para toggle de liveness na aba de configurações."""
        self.liveness_enabled = self.liveness_toggle.get()
        # Salvar automaticamente
        update_liveness_config(self.liveness_enabled, self.liveness_detector.confidence_threshold)

    def _toggle_movement_detection(self):
        """Callback para toggle de detecção de movimento."""
        self.movement_detection_enabled = self.movement_toggle.get()
        # Salvar automaticamente
        update_recognition_config(self.recognition_interval, self.max_distance, self.movement_detection_enabled)

    def _set_frame_size(self, value):
        """Callback para dropdown de tamanho de frame."""
        size_map = {
            "160x120": (120, 160),
            "320x240": (240, 320),
            "640x480": (480, 640),
            "Full HD": None,  # Sem redimensionamento
        }
        self.frame_size = size_map.get(value, (240, 320))
        # Salvar automaticamente
        update_performance_config(value)

    def _reset_stats(self):
        """Reseta estatísticas de liveness."""
        self.liveness_stats = {'checks': 0, 'passed': 0, 'failed': 0}
        # Atualizar display
        self.update_config_stats_display()

    def update_config_stats_display(self):
        """Atualiza as labels de estatísticas na aba de configurações."""
        if not hasattr(self, 'config_checks_label'):
            return
        
        checks = self.liveness_stats['checks']
        passed = self.liveness_stats['passed']
        failed = self.liveness_stats['failed']
        
        self.config_checks_label.configure(text=f"Verificações de Liveness: {checks}")
        self.config_passed_label.configure(text=f"✅ Passou: {passed}")
        self.config_failed_label.configure(text=f"❌ Bloqueado: {failed}")

    def _apply_camera_settings(self):
        """Aplica as configurações de câmera - reinicializa a câmera com novos parâmetros."""
        try:
            # Pegar valores dos dropdowns
            resolution_str = self.resolution_var.get()
            new_width, new_height = map(int, resolution_str.split('x'))
            new_fps = int(self.fps_var.get())
            
            # Fechar câmera antiga
            if self.cap is not None:
                self.cap.release()
                time.sleep(0.5)  # Aguardar liberação
            
            # Reinicializar câmera com novas configurações
            self.cap = cv2.VideoCapture(0)
            
            # ===== NOVA VALIDAÇÃO: Verificar se configuração funcionou =====
            from camera import validate_camera_config
            success, actual_width, actual_height, actual_fps = validate_camera_config(
                self.cap, new_width, new_height, new_fps
            )
            
            # Atualizar variáveis com valores REAIS configurados
            self.camera_width = actual_width
            self.camera_height = actual_height
            self.camera_fps = actual_fps
            
            # ===== SALVAR CONFIGURAÇÕES =====
            update_camera_config(self.supported_resolutions, resolution_str, actual_fps)
            
            # Feedback visual
            if success and (actual_width == new_width and actual_height == new_height):
                # Configuração solicitada funcionou
                self.apply_camera_btn.configure(
                    text=f"✅ Aplicado! ({self.camera_width}x{self.camera_height} @ {self.camera_fps}FPS)",
                    fg_color="#2d7a2d"
                )
                print(f"✅ Câmera configurada: {self.camera_width}x{self.camera_height} @ {self.camera_fps}FPS")
            else:
                # Câmera não suporta, mas conseguiu usar fallback
                self.apply_camera_btn.configure(
                    text=f"⚠️  Usando: {self.camera_width}x{self.camera_height} @ {self.camera_fps}FPS",
                    fg_color="#d4a50a"
                )
                print(f"⚠️  Câmera não suporta {new_width}x{new_height}. Usando {self.camera_width}x{self.camera_height}.")
            
            # Voltar ao texto original após 3 segundos
            self.after(3000, lambda: self.apply_camera_btn.configure(
                text="Aplicar Configurações de Câmera",
                fg_color="#2d7a2d"
            ))
        except Exception as e:
            print(f"❌ Erro ao aplicar configurações de câmera: {e}")
            self.apply_camera_btn.configure(
                text=f"❌ Erro!",
                fg_color="#7a2d2d"
            )
            self.after(2000, lambda: self.apply_camera_btn.configure(
                text="Aplicar Configurações de Câmera",
                fg_color="#2d7a2d"
            ))

    def setup_users_tab(self):
        """Configura a aba de gerenciamento de usuários."""
        main_frame = ctk.CTkFrame(self.users_tab)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Título
        title = ctk.CTkLabel(
            main_frame,
            text="Gerenciamento de Usuários",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        title.pack(pady=(5, 10))

        info = ctk.CTkLabel(
            main_frame,
            text="Clique em um usuário para selecionar",
            font=ctk.CTkFont(size=12),
            text_color="gray",
        )
        info.pack(pady=(0, 10))

        # Frame para scroll e tabela
        users_container = ctk.CTkFrame(main_frame)
        users_container.pack(fill="both", expand=True, padx=0, pady=(0, 10))

        # Criar scrollable frame para usuários
        self.users_scroll_frame = ctk.CTkScrollableFrame(users_container, fg_color="transparent")
        self.users_scroll_frame.pack(fill="both", expand=True)

        # Frame para ações
        actions_frame = ctk.CTkFrame(main_frame)
        actions_frame.pack(fill="x", padx=0, pady=(0, 0))

        # Label da ação selecionada
        self.selected_user_label = ctk.CTkLabel(
            actions_frame,
            text="Nenhum usuário selecionado",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="gray",
        )
        self.selected_user_label.pack(side="left", padx=(0, 20), pady=10)

        # Botão para melhorar cadastro
        self.improve_btn = ctk.CTkButton(
            actions_frame,
            text="📸 Melhorar Cadastro",
            command=self.improve_user_enrollment,
            fg_color="#4a7a2a",
            hover_color="#5a8a3a",
            width=140,
            state="disabled",
        )
        self.improve_btn.pack(side="left", padx=(0, 5))

        # Botão para deletar usuário
        self.delete_btn = ctk.CTkButton(
            actions_frame,
            text="🗑️ Deletar Usuário",
            command=self.delete_user_confirmation,
            fg_color="#7a2a2a",
            hover_color="#8a3a3a",
            width=140,
            state="disabled",
        )
        self.delete_btn.pack(side="left", padx=(0, 5))

        # Botão para atualizar
        refresh_btn = ctk.CTkButton(
            actions_frame,
            text="🔄 Atualizar",
            command=self.refresh_users_list,
            fg_color="#2d5a7a",
            hover_color="#3a6a8a",
            width=100,
        )
        refresh_btn.pack(side="left")

        # Variável para armazenar usuário selecionado
        self.selected_user_var = ctk.StringVar()

        # Carregar lista inicial
        self.refresh_users_list()

    def refresh_users_list(self):
        """Atualiza a lista de usuários com tabela e miniaturas."""
        # Limpar scroll frame anterior
        for widget in self.users_scroll_frame.winfo_children():
            widget.destroy()
        
        users = get_all_users()
        
        if not users:
            empty_label = ctk.CTkLabel(
                self.users_scroll_frame,
                text="Nenhum usuário cadastrado",
                font=ctk.CTkFont(size=12),
                text_color="gray",
            )
            empty_label.pack(pady=20)
            self.selected_user_var.set("")
            self.selected_user_label.configure(text="Nenhum usuário selecionado")
            self.improve_btn.configure(state="disabled")
            self.delete_btn.configure(state="disabled")
            return
        
        # Criar header com colunas
        header_frame = ctk.CTkFrame(self.users_scroll_frame, fg_color="#2a2a2a", height=40)
        header_frame.pack(fill="x", padx=0, pady=(0, 5), ipady=8)
        header_frame.pack_propagate(False)
        
        # Colunas do header
        photo_header = ctk.CTkLabel(header_frame, text="Foto", font=ctk.CTkFont(size=11, weight="bold"), width=80)
        photo_header.pack(side="left", padx=5)
        
        name_header = ctk.CTkLabel(header_frame, text="Nome", font=ctk.CTkFont(size=11, weight="bold"), width=120)
        name_header.pack(side="left", padx=5)
        
        count_header = ctk.CTkLabel(header_frame, text="Fotos", font=ctk.CTkFont(size=11, weight="bold"), width=60)
        count_header.pack(side="left", padx=5)
        
        actions_header = ctk.CTkLabel(header_frame, text="Ações", font=ctk.CTkFont(size=11, weight="bold"), width=140)
        actions_header.pack(side="left", padx=5)
        
        # Armazenar referências aos frames das linhas para seleção
        self.user_row_widgets = {}
        
        # Criar linhas da tabela
        for i, (username, num_embeddings) in enumerate(users):
            row_frame = ctk.CTkFrame(
                self.users_scroll_frame,
                fg_color="#1a1a1a" if i % 2 == 0 else "#252525",
                corner_radius=4,
                height=80,
            )
            row_frame.pack(fill="x", padx=0, pady=3, ipady=8)
            row_frame.pack_propagate(False)
            
            # Guardar referência e username para seleção
            row_frame.username = username
            
            # Carregar e exibir miniatura
            thumbnail_label = ctk.CTkLabel(row_frame, text="", width=70, height=70)
            thumbnail_label.pack(side="left", padx=5, pady=5)
            
            # Carregar thumbnail se existir
            thumbnail_path = f"database/users/{username}/thumbnail.jpg"
            if os.path.exists(thumbnail_path):
                try:
                    img = Image.open(thumbnail_path)
                    img_tk = ImageTk.PhotoImage(img)
                    thumbnail_label.configure(image=img_tk, text="")
                    thumbnail_label.image = img_tk  # Manter referência
                except Exception as e:
                    thumbnail_label.configure(text="❌", text_color="gray")
                    print(f"Erro ao carregar thumbnail de {username}: {e}")
            else:
                # Placeholder se não houver thumbnail
                thumbnail_label.configure(text="📷", text_color="gray")
            
            # Nome do usuário
            name_label = ctk.CTkLabel(
                row_frame,
                text=username,
                font=ctk.CTkFont(size=12, weight="bold"),
                width=120,
                anchor="w",
            )
            name_label.pack(side="left", padx=5)
            
            # Número de fotos
            count_label = ctk.CTkLabel(
                row_frame,
                text=str(num_embeddings),
                font=ctk.CTkFont(size=12),
                width=60,
                anchor="center",
                text_color="gray",
            )
            count_label.pack(side="left", padx=5)
            
            # Botões de ação
            actions_frame = ctk.CTkFrame(row_frame, fg_color="transparent", width=140)
            actions_frame.pack(side="left", padx=5)
            actions_frame.pack_propagate(False)
            
            # Botão melhorar
            improve_btn_small = ctk.CTkButton(
                actions_frame,
                text="📸",
                font=ctk.CTkFont(size=14),
                width=35,
                height=30,
                fg_color="#4a7a2a",
                hover_color="#5a8a3a",
                command=lambda u=username: self.on_user_row_click(u, "improve"),
            )
            improve_btn_small.pack(side="left", padx=2)
            
            # Botão deletar
            delete_btn_small = ctk.CTkButton(
                actions_frame,
                text="🗑️",
                font=ctk.CTkFont(size=14),
                width=35,
                height=30,
                fg_color="#7a2a2a",
                hover_color="#8a3a3a",
                command=lambda u=username: self.on_user_row_click(u, "delete"),
            )
            delete_btn_small.pack(side="left", padx=2)
            
            # Adicionar binding para seleção ao clicar na linha
            row_frame.bind("<Button-1>", lambda e, u=username: self.on_user_row_click(u, "select"))
            name_label.bind("<Button-1>", lambda e, u=username: self.on_user_row_click(u, "select"))
            count_label.bind("<Button-1>", lambda e, u=username: self.on_user_row_click(u, "select"))
            thumbnail_label.bind("<Button-1>", lambda e, u=username: self.on_user_row_click(u, "select"))
            
            # Guardar referência para highlighting
            self.user_row_widgets[username] = row_frame
        
        # Atualizar estado dos botões
        if self.selected_user_var.get():
            self.improve_btn.configure(state="normal")
            self.delete_btn.configure(state="normal")
        else:
            self.improve_btn.configure(state="disabled")
            self.delete_btn.configure(state="disabled")

    def on_user_row_click(self, username, action):
        """Callback quando um usuário é clicado na tabela."""
        if action == "select":
            # Selecionar usuário
            self.selected_user_var.set(username)
            self.selected_user_label.configure(text=f"Selecionado: {username}")
            
            # Atualizar cor de seleção
            for u, row in self.user_row_widgets.items():
                if u == username:
                    row.configure(fg_color="#2a4a2a")  # Verde escuro
                else:
                    idx = list(self.user_row_widgets.keys()).index(u)
                    row.configure(fg_color="#1a1a1a" if idx % 2 == 0 else "#252525")
            
            # Ativar botões
            self.improve_btn.configure(state="normal")
            self.delete_btn.configure(state="normal")
        
        elif action == "improve":
            self.selected_user_var.set(username)
            self.improve_user_enrollment()
        
        elif action == "delete":
            self.selected_user_var.set(username)
            self.delete_user_confirmation()

    def improve_user_enrollment(self):
        """Abre janela nova para adicionar mais fotos a um usuário."""
        selected_user = self.selected_user_var.get()
        
        if not selected_user:
            self.show_error("Selecione um usuário para melhorar o cadastro")
            return
        
        # Abre janela nova para captura
        self.open_improve_enrollment_window(selected_user)

    def open_improve_enrollment_window(self, username):
        """Abre uma janela dedicada para melhorar o cadastro de um usuário."""
        improve_window = ctk.CTkToplevel(self)
        improve_window.title(f"Melhorar Cadastro - {username}")
        improve_window.geometry("1000x700")
        improve_window.resizable(True, True)
        
        # Variáveis locais para a janela
        window_data = {
            'username': username,
            'capture_event': threading.Event(),
            'current_frame': None,
            'frame_lock': threading.Lock(),
            'is_running': True,
            'instruction_index': 0,
            'captured_frames': []
        }

        # Frame principal
        main_frame = ctk.CTkFrame(improve_window)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Lado esquerdo - Controles
        left_frame = ctk.CTkFrame(main_frame)
        left_frame.pack(side="left", fill="both", padx=(0, 10), pady=10)

        # Título
        title_label = ctk.CTkLabel(
            left_frame, 
            text=f"Melhorar Cadastro\n{username}", 
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title_label.pack(pady=(10, 20))

        # Instruções
        instructions_list = [
            "Olhe para frente",
            "Vire a cabeça para a direita",
            "Vire a cabeça para a esquerda",
            "Olhe para cima",
            "Olhe para baixo",
            "Sorria ou expressão neutra",
        ]

        instructions_label = ctk.CTkLabel(
            left_frame, 
            text="Instruções:", 
            font=ctk.CTkFont(size=14, weight="bold")
        )
        instructions_label.pack(pady=(0, 10))

        instruction_textbox = ctk.CTkTextbox(
            left_frame, 
            width=250, 
            height=150, 
            wrap="word"
        )
        instruction_textbox.pack(pady=(0, 20))
        
        instructions_text = "\n".join([f"{i+1}. {instr}" for i, instr in enumerate(instructions_list)])
        instruction_textbox.insert("1.0", instructions_text)
        instruction_textbox.configure(state="disabled")

        # Status do cadastro
        status_label = ctk.CTkLabel(
            left_frame, 
            text="Status:", 
            font=ctk.CTkFont(size=12, weight="bold")
        )
        status_label.pack(pady=(10, 5))

        status_display = ctk.CTkLabel(
            left_frame, 
            text="Aguardando...", 
            font=ctk.CTkFont(size=12), 
            text_color="yellow"
        )
        status_display.pack(pady=(0, 20))

        # Progresso
        progress_bar = ctk.CTkProgressBar(left_frame, width=250)
        progress_bar.pack(pady=(0, 20))
        progress_bar.set(0)

        # Botão de capturar
        capture_button = ctk.CTkButton(
            left_frame,
            text="📷 Capturar Foto",
            command=lambda: window_data['capture_event'].set(),
            width=250,
            height=50,
            font=ctk.CTkFont(size=18, weight="bold"),
            fg_color="green",
            hover_color="darkgreen",
        )
        capture_button.pack(pady=(0, 10))

        # Botão de fechar
        close_button = ctk.CTkButton(
            left_frame,
            text="❌ Fechar",
            command=lambda: self.close_improve_enrollment(improve_window, window_data),
            width=250,
            height=40,
            font=ctk.CTkFont(size=14),
            fg_color="#7a2a2a",
            hover_color="#8a3a3a",
        )
        close_button.pack(pady=(0, 10))

        # Lado direito - Preview
        right_frame = ctk.CTkFrame(main_frame)
        right_frame.pack(side="right", fill="both", expand=True, pady=10)

        camera_label = ctk.CTkLabel(
            right_frame, 
            text="Preview da Câmera", 
            font=ctk.CTkFont(size=16)
        )
        camera_label.pack(pady=(10, 5))

        video_label = ctk.CTkLabel(right_frame, text="")
        video_label.pack(pady=10, padx=10, expand=True, fill="both")

        # Iniciar threads
        def get_current_frame():
            with window_data['frame_lock']:
                return window_data['current_frame'].copy() if window_data['current_frame'] is not None else None

        def update_progress(value):
            improve_window.after(0, lambda: progress_bar.set(value))

        def update_status(text, color="white"):
            improve_window.after(0, lambda: status_display.configure(text=text, text_color=color))

        # Inicia thread de captura
        capture_thread = threading.Thread(
            target=self._improve_enrollment_capture_thread,
            args=(window_data,),
            daemon=True
        )
        capture_thread.start()

        # Inicia thread de preview
        preview_thread = threading.Thread(
            target=self._improve_enrollment_preview_thread,
            args=(window_data, video_label),
            daemon=True
        )
        preview_thread.start()

        # Inicia thread de processamento
        process_thread = threading.Thread(
            target=self._improve_enrollment_process_thread,
            args=(window_data, instructions_list, update_progress, update_status),
            daemon=True
        )
        process_thread.start()

        def on_closing():
            window_data['is_running'] = False
            self.reload_database()
            improve_window.destroy()

        improve_window.protocol("WM_DELETE_WINDOW", on_closing)

    def _improve_enrollment_capture_thread(self, window_data):
        """Thread para capturar frames da câmera durante melhoria."""
        while window_data['is_running']:
            if self.cap is None:
                time.sleep(0.05)
                continue

            try:
                ret, frame = self.cap.read()
                if ret and frame is not None and frame.size > 0:
                    if frame.shape[0] > 0 and frame.shape[1] > 0:
                        frame_resized = cv2.resize(frame, (320, 240))
                        with window_data['frame_lock']:
                            window_data['current_frame'] = frame_resized.copy()
            except Exception as e:
                time.sleep(0.05)
                continue

            time.sleep(0.03)  # ~30 FPS

    def _improve_enrollment_preview_thread(self, window_data, video_label):
        """Thread para atualizar preview da câmera."""
        while window_data['is_running']:
            with window_data['frame_lock']:
                if window_data['current_frame'] is not None:
                    frame = window_data['current_frame'].copy()
                else:
                    frame = None

            if frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img = img.resize((320, 240), Image.Resampling.LANCZOS)
                img_tk = ImageTk.PhotoImage(image=img)

                video_label.configure(image=img_tk)
                video_label.image = img_tk

            time.sleep(0.03)

    def _improve_enrollment_process_thread(self, window_data, instructions, update_progress, update_status):
        """Thread para processar o fluxo de captura e embedding."""
        try:
            print("[DEBUG] Iniciando _improve_enrollment_process_thread")
            
            user_info = get_user_info(window_data['username'])
            if not user_info:
                update_status("Erro ao carregar informacoes do usuario", "red")
                print(f"[Erro] user_info vazio para {window_data['username']}")
                return
            
            next_index = max(user_info['embeddings']) + 1 if user_info['embeddings'] else 0
            print(f"[DEBUG] next_index = {next_index}")
            
            update_status(f"Iniciando melhoria de cadastro ({next_index} fotos existentes)", "blue")
            
            captured_count = 0
            total_steps = len(instructions) * 2  # Captura + processamento
            
            # Fase 1: Captura de frames
            for i, instr in enumerate(instructions):
                if not window_data['is_running']:
                    print(f"[DEBUG] Abortando captura: is_running=False")
                    break
                
                update_status(f"Passo {i+1}/{len(instructions)}: {instr}\nClique em 'Capturar Foto' quando estiver pronto", "yellow")
                update_progress(i / total_steps)
                
                # Aguardar clique no botão
                window_data['capture_event'].clear()
                captured = window_data['capture_event'].wait(timeout=120)  # Timeout de 2 min
                
                if not window_data['is_running']:
                    print(f"[DEBUG] Abortando captura apos evento: is_running=False")
                    break
                
                if not captured:
                    update_status(f"Timeout na captura do passo {i+1}", "red")
                    print(f"[DEBUG] Timeout na captura do passo {i+1}")
                    continue
                
                # Capturar frame
                try:
                    with window_data['frame_lock']:
                        if window_data['current_frame'] is not None:
                            frame = window_data['current_frame'].copy()
                        else:
                            frame = None
                except Exception as e:
                    print(f"[Erro] Ao copiar frame: {e}")
                    frame = None
                
                if frame is None:
                    update_status(f"Falha na captura do passo {i+1} (frame vazio)", "red")
                    print(f"[DEBUG] Frame None no passo {i+1}")
                    continue
                
                # Validar se há rosto
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                except Exception as e:
                    print(f"[Erro] Ao detectar rosto: {e}")
                    update_status(f"Erro ao detectar rosto no passo {i+1}", "red")
                    continue
                
                if len(faces) == 0:
                    update_status(f"Rosto nao detectado no passo {i+1}.\nTente novamente", "red")
                    print(f"[DEBUG] Nenhum rosto detectado no passo {i+1}")
                    continue
                
                try:
                    window_data['captured_frames'].append(frame.copy())
                    update_status(f"OK Foto {i+1} capturada!", "green")
                    print(f"[DEBUG] Foto {i+1} capturada com sucesso")
                except Exception as e:
                    print(f"[Erro] Ao adicionar frame a lista: {e}")
                    continue
                
                time.sleep(0.5)
            
            print(f"[DEBUG] Fase 1 completa. Frames capturados: {len(window_data['captured_frames'])}")
            
            if len(window_data['captured_frames']) == 0:
                update_status("Nenhum frame capturado!", "red")
                print("[Erro] Nenhum frame foi capturado")
                return
            
            # Fase 2: Processamento de embeddings
            update_status("Processando embeddings...", "blue")
            print(f"[DEBUG] Iniciando processamento de {len(window_data['captured_frames'])} frames")
            
            saved_count = 0
            new_embs = []
            
            for i, frame in enumerate(window_data['captured_frames']):
                if not window_data['is_running']:
                    print(f"[DEBUG] Abortando processamento no frame {i}: is_running=False")
                    break
                
                update_progress((len(instructions) + i) / total_steps)
                
                try:
                    print(f"[DEBUG] Processando frame {i+1}/{len(window_data['captured_frames'])}")
                    emb = get_embedding(frame)
                    new_embs.append(emb)
                    
                    # Criar diretorio se nao existir
                    user_dir = f"database/users/{window_data['username']}"
                    os.makedirs(user_dir, exist_ok=True)
                    
                    save_path = f"{user_dir}/{next_index + saved_count}.npy"
                    np.save(save_path, emb)
                    print(f"[DEBUG] Embedding salvo: {save_path}")
                    
                    saved_count += 1
                    update_status(f"Processando... {saved_count}/{len(window_data['captured_frames'])}", "blue")
                except Exception as e:
                    print(f"[Erro] Ao processar frame {i}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            print(f"[DEBUG] Fase 2 completa. Embeddings salvos: {saved_count}")
            update_progress(1.0)
            
            if saved_count > 0:
                print(f"[DEBUG] Tentando atualizar banco em memoria e UI")
                try:
                    # Atualizar banco em memória - PROTEGIDO
                    if window_data['is_running']:  # Verificar se janela ainda existe
                        self.after(0, lambda: self.append_new_embeddings(window_data['username'], np.array(new_embs)))
                        self.after(0, lambda: self.refresh_users_list())
                        update_status(f"OK Cadastro melhorado! {saved_count} fotos adicionadas", "green")
                        print(f"[Sucesso] {saved_count} embeddings salvos para {window_data['username']}")
                except Exception as e:
                    print(f"[Erro] Ao atualizar UI: {str(e)}")
                    import traceback
                    traceback.print_exc()
            else:
                update_status("Erro ao processar embeddings", "red")
                print("[Erro] Nenhum embedding foi processado com sucesso")
        
        except Exception as e:
            print(f"[Erro Fatal] Melhoria de cadastro: {str(e)}")
            import traceback
            traceback.print_exc()
            try:
                update_status(f"Erro: {str(e)}", "red")
            except:
                pass

    def close_improve_enrollment(self, window, window_data):
        """Fecha a janela de melhoria de cadastro."""
        window_data['is_running'] = False
        window.destroy()


    def delete_user_confirmation(self):
        """Mostra diálogo de confirmação antes de deletar."""
        selected_user = self.selected_user_var.get()
        
        if not selected_user:
            self.show_error("Selecione um usuário para deletar")
            return
        
        # Criar diálogo de confirmação
        dialog = ctk.CTkToplevel(self)
        dialog.title("Confirmar Deleção")
        dialog.geometry("400x150")
        dialog.resizable(False, False)
        
        # Trazer para frente
        dialog.attributes("-topmost", True)
        
        # Mensagem
        msg = ctk.CTkLabel(
            dialog,
            text=f"Tem certeza que deseja deletar '{selected_user}'?\n\nEsta ação não pode ser desfeita.",
            font=ctk.CTkFont(size=12),
            wraplength=350,
        )
        msg.pack(pady=(20, 15))
        
        # Botões
        buttons_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        buttons_frame.pack(pady=(0, 20))
        
        # Botão Cancelar
        cancel_btn = ctk.CTkButton(
            buttons_frame,
            text="❌ Cancelar",
            command=dialog.destroy,
            fg_color="#4a4a4a",
            hover_color="#5a5a5a",
            width=100,
        )
        cancel_btn.pack(side="left", padx=5)
        
        # Botão Deletar
        def confirm_delete():
            success, message = delete_user(selected_user)
            dialog.destroy()
            
            if success:
                print(message)
                self.selected_user_var.set("")
                self.refresh_users_list()
                self.reload_database()
                self.show_success(message)
            else:
                self.show_error(message)
        
        delete_btn = ctk.CTkButton(
            buttons_frame,
            text="🗑️ Deletar",
            command=confirm_delete,
            fg_color="#7a2a2a",
            hover_color="#8a3a3a",
            width=100,
        )
        delete_btn.pack(side="left", padx=5)

    def show_success(self, message):
        """Mostra mensagem de sucesso."""
        print(f"✅ {message}")
        # Pode ser expandido para mostrar notificação visual

    def setup_log_tab(self):
        """Configura a aba de logs (entradas do dia)."""
        main_frame = ctk.CTkFrame(self.log_tab)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        title = ctk.CTkLabel(
            main_frame,
            text="Log de entradas (hoje)",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        title.pack(pady=(5, 10))

        info = ctk.CTkLabel(
            main_frame,
            text="Aqui aparecem as pessoas reconhecidas, com data e horário.",
            font=ctk.CTkFont(size=12),
            text_color="gray",
        )
        info.pack(pady=(0, 10))

        self.log_textbox = ctk.CTkTextbox(
            main_frame,
            width=800,
            height=300,
            wrap="none",
        )
        self.log_textbox.pack(fill="both", expand=False, padx=5, pady=(0, 10))
        self.log_textbox.configure(state="disabled")

        # ===== FRAME DE INFORMAÇÕES EM TEMPO REAL =====
        info_frame = ctk.CTkFrame(main_frame)
        info_frame.pack(fill="x", padx=5, pady=(10, 0))

        # Resultado do reconhecimento
        result_label = ctk.CTkLabel(
            info_frame,
            text="Resultado:",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="gray"
        )
        result_label.pack(side="left", padx=(0, 10))

        self.result_display = ctk.CTkLabel(
            info_frame,
            text="Aguardando...",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="white"
        )
        self.result_display.pack(side="left", padx=(0, 20))

        # Status de Liveness
        liveness_label = ctk.CTkLabel(
            info_frame,
            text="Liveness:",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="gray"
        )
        liveness_label.pack(side="left", padx=(0, 10))

        self.liveness_display = ctk.CTkLabel(
            info_frame,
            text="🔒 ON",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="green"
        )
        self.liveness_display.pack(side="left", padx=(0, 20))

        # Estatísticas
        stats_label = ctk.CTkLabel(
            info_frame,
            text="Checks/Passed:",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="gray"
        )
        stats_label.pack(side="left", padx=(0, 10))

        self.stats_display = ctk.CTkLabel(
            info_frame,
            text="0/0",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="white"
        )
        self.stats_display.pack(side="left")

    def start_enrollment(self):
        """Inicia o processo de cadastro"""
        name = self.name_entry.get().strip()
        if not name:
            self.show_error("Por favor, informe o nome do usuário")
            return

        if self.is_enrollment_running:
            return

        self.is_enrollment_running = True
        self.enroll_button.configure(state="disabled")
        self.name_entry.configure(state="disabled")
        self.capture_event.clear()  # Limpar evento anterior
        
        # Mostrar botão de capturar
        self.capture_button.pack(pady=(0, 10))
        self.enroll_status.configure(text="Aguardando primeira foto...", text_color="yellow")

        # Executar cadastro em thread separada
        thread = threading.Thread(target=self.enroll_thread, args=(name,), daemon=True)
        thread.start()
    
    def capture_photo(self):
        """Captura uma foto manualmente quando o botão é clicado"""
        self.capture_event.set()  # Sinaliza para a thread de cadastro capturar

    def enroll_thread(self, user_name):
        """Thread para executar o cadastro"""
        try:
            instructions = [
                "Olhe para frente",
                "Vire a cabeça para a direita",
                "Vire a cabeça para a esquerda",
                "Olhe para cima",
                "Olhe para baixo",
                "Sorria ou expressão neutra",
            ]

            new_embs = guided_enroll_gui_manual(
                self.cap,
                user_name,
                face_cascade,
                instructions,
                self.capture_event,
                self.update_enroll_progress,
                self.update_enroll_status,
                self.get_current_enroll_frame,
                self.save_first_photo_as_thumbnail,  # Callback para salvar primeira foto
            )

            # Atualiza banco em memória de forma incremental
            if new_embs is not None and new_embs.size > 0:
                self.after(0, lambda: self.append_new_embeddings(user_name, new_embs))

            self.after(0, self.enroll_complete)
        except Exception as e:
            self.after(0, lambda: self.enroll_error(str(e)))
    
    def get_current_enroll_frame(self):
        """Retorna o frame atual da câmera para cadastro"""
        with self.enroll_frame_lock:
            return self.current_enroll_frame.copy() if self.current_enroll_frame is not None else None
    
    def save_first_photo_as_thumbnail(self, username, frame):
        """Salva a primeira foto como miniatura JPG para exibição na lista de usuários"""
        try:
            user_dir = f"database/users/{username}"
            os.makedirs(user_dir, exist_ok=True)
            
            # Salvar como JPG com tamanho reduzido
            thumbnail_path = os.path.join(user_dir, "thumbnail.jpg")
            
            # Redimensionar para 150x150 para miniatura
            thumbnail = cv2.resize(frame, (150, 150))
            cv2.imwrite(thumbnail_path, thumbnail)
            
            print(f"✅ Miniatura salva: {thumbnail_path}")
        except Exception as e:
            print(f"⚠️ Erro ao salvar miniatura: {e}")
    
    def start_enroll_preview(self):
        """Inicia o preview da câmera na aba de cadastro"""
        self.enroll_preview_running = True
        thread = threading.Thread(target=self.enroll_preview_thread, daemon=True)
        thread.start()
    
    def enroll_preview_thread(self):
        """Thread para atualizar o preview da câmera na aba de cadastro"""
        while True:
            if self.cap is None:
                time.sleep(0.1)
                continue

            # Se o reconhecimento estiver rodando, não disputar a câmera
            if self.is_recognition_running:
                time.sleep(0.1)
                continue
            
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None and frame.size > 0:
                    # Validar se frame é válido
                    if frame.shape[0] > 0 and frame.shape[1] > 0:
                        with self.enroll_frame_lock:
                            self.current_enroll_frame = frame.copy()
                        
                        # Sempre atualizar preview (em tempo real)
                        self.after(0, lambda f=frame: self.display_frame_enroll(f))
            except cv2.error as e:
                # Frame inválido ou erro na câmera - silenciar e continuar
                time.sleep(0.1)
                continue
            except Exception as e:
                # Erro geral
                time.sleep(0.1)
                continue
            
            time.sleep(0.03)  # ~30 FPS

    def update_enroll_progress(self, value):
        """Atualiza a barra de progresso do cadastro"""
        self.after(0, lambda: self.enroll_progress.set(value))

    def update_enroll_status(self, text, color="white"):
        """Atualiza o status do cadastro"""
        # Extrair a instrução do texto (primeira linha antes de \\n)
        first_line = text.split('\n')[0] if text else ""
        
        # Se contiver uma instrução de pose (ex: "Passo 1/6: Olhe para frente")
        if "Passo" in first_line and ":" in first_line:
            # Extrai a instrução após o ':'
            instruction_part = first_line.split(":", 1)[1].strip()
            self.current_enrollment_instruction = instruction_part
        else:
            self.current_enrollment_instruction = first_line
        
        self.after(0, lambda: self.enroll_status.configure(text=text, text_color=color))

    def update_enroll_frame(self, frame):
        """Atualiza o frame de vídeo do cadastro"""
        self.after(0, lambda: self.display_frame_enroll(frame))

    def display_frame_enroll(self, frame):
        """Exibe o frame na aba de cadastro"""
        if frame is None:
            return

        # Redimensionar frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = img.resize((self.camera_width, self.camera_height), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(image=img)

        self.enroll_video_label.configure(image=img_tk)
        self.enroll_video_label.image = img_tk

    def enroll_complete(self):
        """Callback quando o cadastro é concluído"""
        self.is_enrollment_running = False
        self.enroll_button.configure(state="normal")
        self.name_entry.configure(state="normal")
        self.capture_button.pack_forget()  # Ocultar botão de capturar
        self.enroll_status.configure(text="Cadastro concluído com sucesso!", text_color="green")
        self.enroll_progress.set(0)
        self.name_entry.delete(0, "end")

    def append_new_embeddings(self, user_name, new_embs):
        """Atualiza o banco em memória apenas com os novos embeddings do cadastro."""
        try:
            if new_embs is None or new_embs.size == 0:
                return

            # Garante formato (N, D)
            if new_embs.ndim == 1:
                new_embs = new_embs.reshape(1, -1)

            count = new_embs.shape[0]
            new_labels = np.array([user_name] * count)

            if self.db is None or "embs" not in self.db or self.db["embs"].size == 0:
                # Primeiro cadastro em memória
                self.db = {
                    "embs": new_embs.copy(),
                    "labels": new_labels,
                }
            else:
                self.db["embs"] = np.vstack([self.db["embs"], new_embs])
                self.db["labels"] = np.concatenate([self.db["labels"], new_labels])

            # Feedback visual rápido
            self.recognize_status.configure(
                text=f"Banco atualizado (+{count} embeddings)", text_color="blue"
            )
            self.after(
                2000,
                lambda: self.recognize_status.configure(
                    text="Pronto", text_color="gray"
                ),
            )
        except Exception as e:
            print(f"Erro ao atualizar banco em memória: {e}")

    def enroll_error(self, error_msg):
        """Callback quando há erro no cadastro"""
        self.is_enrollment_running = False
        self.enroll_button.configure(state="normal")
        self.name_entry.configure(state="normal")
        self.capture_button.pack_forget()  # Ocultar botão de capturar
        self.enroll_status.configure(text=f"Erro: {error_msg}", text_color="red")
        self.enroll_progress.set(0)

    def toggle_recognition(self):
        """Inicia ou para o reconhecimento"""
        if not self.is_recognition_running:
            self.start_recognition()
        else:
            self.stop_recognition()

    def start_recognition(self):
        """Inicia o reconhecimento"""
        if self.db is None:
            self.reload_database()

        if self.db["embs"].size == 0:
            self.show_error("Banco de dados vazio. Faça cadastros primeiro.")
            return

        self.is_recognition_running = True
        self.recognize_button.configure(text="Parar Reconhecimento")
        self.recognize_status.configure(text="Reconhecendo...", text_color="green")

        # Threads de captura e reconhecimento
        self.capture_thread = threading.Thread(target=self.capture_thread_func, daemon=True)
        self.recognition_thread = threading.Thread(
            target=self.recognition_thread_func, daemon=True
        )
        self.capture_thread.start()
        self.recognition_thread.start()
        
        # Atualizar informações da aba de Logs
        self.update_recognition_info()

    def update_recognition_info(self):
        """Atualiza as informações da aba de Logs em tempo real"""
        if not self.is_recognition_running:
            return

        with self.frame_lock:
            result = self.shared_result
        
        # ===== ATUALIZAR LABELS CustomTkinter =====
        # Resultado principal
        if "Acesso bloqueado" in result or "Spoofing" in result:
            result_color = "red"
            display_result = f"❌ {result}"
        elif "Não reconhecido" in result or "Aguardando" in result:
            result_color = "orange"
            display_result = f"⚪ {result}"
        else:
            result_color = "green"
            display_result = f"✅ {result}"
        
        if hasattr(self, 'result_display'):
            self.result_display.configure(text=display_result, text_color=result_color)

        # Status de Liveness
        if hasattr(self, 'liveness_display'):
            if self.liveness_enabled:
                self.liveness_display.configure(text="🔒 ON", text_color="green")
            else:
                self.liveness_display.configure(text="🔓 OFF", text_color="gray")

        # Estatísticas
        if hasattr(self, 'stats_display'):
            checks = self.liveness_stats['checks']
            passed = self.liveness_stats['passed']
            self.stats_display.configure(text=f"{checks}/{passed}")

        # Agendar próxima atualização
        self.after(100, self.update_recognition_info)  # 100ms é suficiente para atualizar labels

    def stop_recognition(self):
        """Para o reconhecimento e mostra estatísticas de otimização e liveness"""
        self.is_recognition_running = False
        self.recognize_button.configure(text="Iniciar Reconhecimento")
        self.recognize_status.configure(text="Parado", text_color="gray")
        self.shared_result = "Não reconhecido"
        
        # ===== OTIMIZAÇÃO: Exibir estatísticas =====
        if self.recognition_count > 0:
            print(f"[GUI Otimizado] Reconhecimentos: {self.recognition_count} | Pulados: {self.skipped_count}")
        
        # ===== SEGURANÇA: Exibir estatísticas de liveness =====
        if self.liveness_stats['checks'] > 0:
            passed = self.liveness_stats['passed']
            failed = self.liveness_stats['failed']
            total = self.liveness_stats['checks']
            pass_rate = (passed / total * 100) if total > 0 else 0
            print(f"[Liveness] Total: {total} | Passou: {passed} ({pass_rate:.1f}%) | Falhou: {failed}")
        
        self.recognition_count = 0
        self.skipped_count = 0
        self.prev_frame_recognition = None
        self.liveness_stats = {'checks': 0, 'passed': 0, 'failed': 0}

    def toggle_liveness(self):
        """Alterna entre validação de liveness ON/OFF"""
        self.liveness_enabled = self.liveness_toggle.get()
        status = "🔒 ATIVADO" if self.liveness_enabled else "🔓 DESATIVADO"
        print(f"[Liveness] Validação de Liveness: {status}")
        
        # Salvar configuração
        update_liveness_config(self.liveness_enabled, self.liveness_detector.confidence_threshold)
        
        # Resetar estatísticas quando alterna
        self.liveness_stats = {'checks': 0, 'passed': 0, 'failed': 0}

    def set_liveness_threshold(self, val):
        """Callback do slider para atualizar o threshold do detector em tempo real."""
        try:
            # `val` pode ser string (CustomTkinter envia float); garantir float
            v = float(val)
        except Exception:
            return

        # Atualiza o detector (se existir)
        if hasattr(self, 'liveness_detector') and self.liveness_detector is not None:
            try:
                self.liveness_detector.confidence_threshold = v
            except Exception:
                pass

        # Atualiza label visual
        if hasattr(self, 'liveness_threshold_label'):
            self.liveness_threshold_label.configure(text=f"Threshold: {v:.2f}")
        
        # Salvar configuração
        update_liveness_config(self.liveness_enabled, v)

    def capture_thread_func(self):
        """Thread de captura de frames com redimensionamento otimizado"""
        while self.is_recognition_running:
            if self.cap is None:
                time.sleep(0.1)
                continue
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None and frame.size > 0:
                    # Validar se frame é válido antes de processar
                    if frame.shape[0] > 0 and frame.shape[1] > 0:
                        # ===== OTIMIZAÇÃO #4: Redimensionar para 320x240 =====
                        frame_resized = cv2.resize(frame, (320, 240))
                        with self.frame_lock:
                            self.shared_frame = frame_resized.copy()
            except cv2.error as e:
                # Frame inválido ou erro na câmera - silenciar e continuar
                time.sleep(0.1)
                continue
            except Exception as e:
                print(f"[Erro] capture_thread_func: {e}")
                time.sleep(0.1)
                continue
            time.sleep(0.03)  # ~30 FPS

    def recognition_thread_func(self):
        """
        Thread de reconhecimento com otimizações e validação de liveness:
        - Intervalo reduzido: 500ms → 150ms
        - Detecção de movimento para pular frames similares
        - Validação de liveness ANTES do reconhecimento
        - Uma detecção Haar apenas
        """
        # ===== OTIMIZAÇÃO #2: Intervalo reduzido - usando configuração ajustável =====
        last_time = 0

        while self.is_recognition_running:
            now = time.time()
            if now - last_time < self.recognition_interval:
                time.sleep(0.01)
                continue
            last_time = now

            with self.frame_lock:
                if self.shared_frame is None:
                    continue
                frame = self.shared_frame.copy()

            # ===== OTIMIZAÇÃO #2: Detecção de movimento para pular frames =====
            if self.movement_detection_enabled:
                similarity = _frame_similarity(self.prev_frame_recognition, frame)
                if similarity > 0.98:
                    self.skipped_count += 1
                    self.prev_frame_recognition = frame.copy()
                    continue  # Pula reconhecimento se frames são muito similares
            
            self.prev_frame_recognition = frame.copy()

            # ===== OTIMIZAÇÃO #3: Detectar rosto uma única vez aqui =====
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) == 0:
                with self.frame_lock:
                    self.shared_result = "Nao reconhecido"
                continue

            # ===== SEGURANÇA: Validação de liveness ANTES do reconhecimento =====
            liveness_ok = True
            liveness_msg = ""
            
            if self.liveness_enabled:
                self.liveness_stats['checks'] += 1
                # Usar o primeiro rosto detectado para validação de liveness
                face = faces[0]
                liveness_check = self.liveness_detector.validate_frame(frame, face)
                
                if liveness_check['is_live']:
                    self.liveness_stats['passed'] += 1
                    quality = liveness_check.get('quality_ok', True)
                    liveness_msg = f"Vivo"
                else:
                    self.liveness_stats['failed'] += 1
                    liveness_msg = f""
                    liveness_ok = False
                
                # Atualizar display das estatísticas na aba de configurações
                self.after(0, self.update_config_stats_display)
            
            # Reconhecer apenas se passou em liveness
            if liveness_ok:
                try:
                    if self.liveness_enabled:
                        # Usar reconhecimento com liveness integrado
                        # recognize_with_liveness requer: frame, face, db, detector
                        result = recognize_with_liveness(frame, faces[0], self.db, self.liveness_detector)
                        user = result['user']
                        dist = result['distance']
                        allowed = result['allowed']
                    else:
                        # Reconhecimento padrão
                        user, dist = recognize(frame, self.db)
                        allowed = True
                    
                    self.recognition_count += 1
                    
                    if user:
                        status_text = f"{user} ({dist:.2f})"
                        
                        # Indicar se foi bloqueado por liveness
                        if self.liveness_enabled and not allowed:
                            status_text += " [BLOQUEADO]"
                        
                        # Adicionar status de liveness se ativado
                        if self.liveness_enabled and liveness_msg:
                            status_text += f" | {liveness_msg}"
                        
                        with self.frame_lock:
                            self.shared_result = status_text
                        
                        # Atualiza última pessoa reconhecida (não some quando perde o rosto)
                        self.last_recognized_user = user
                        self.after(0, self.update_last_recognized_label)
                        # Registra log de reconhecimento
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        log_msg = f"[{timestamp}] {user} ({dist:.2f})"
                        if self.liveness_enabled:
                            log_msg += f" | Liveness: {'✅' if allowed else '❌'}"
                        print(log_msg)
                        self.after(0, lambda u=user, d=dist: self.log_recognition(u, d))
                    else:
                        with self.frame_lock:
                            self.shared_result = "Nao reconhecido"
                except Exception as e:
                    print(f"Erro no reconhecimento: {e}")
            else:
                # Liveness falhou - bloquear acesso
                with self.frame_lock:
                    self.shared_result = liveness_msg if liveness_msg else "Acesso bloqueado"

    def open_client_window(self):
        """Abre uma segunda janela para a tela do cliente, usando o mesmo vídeo"""
        # Se já existir e ainda estiver aberta, apenas traz para frente
        if self.client_window is not None and self.client_window.winfo_exists():
            self.client_window.lift()
            return

        self.client_window = ctk.CTkToplevel(self)
        self.client_window.title("Tela do Cliente")
        self.client_window.geometry("900x700")
        self.client_window.configure(fg_color="black")

        # Janela do cliente normalmente vai para a outra tela (monitor externo)
        info_label = ctk.CTkLabel(
            self.client_window,
            text="Tela do cliente\n(posicione esta janela no monitor voltado para o cliente)",
            font=ctk.CTkFont(size=14),
            text_color="gray",
        )
        info_label.pack(pady=(10, 5))

        # ===== Status em tempo real (em vez de "Última pessoa reconhecida") =====
        # Colocar o status **acima** do vídeo para garantir visibilidade
        self.client_status_label = ctk.CTkLabel(
            self.client_window,
            text="Aguardando...",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="white",
        )
        self.client_status_label.pack(pady=(8, 6))

        self.client_video_label = ctk.CTkLabel(self.client_window, text="")
        self.client_video_label.pack(pady=6, padx=10, expand=True)

        # Começa a atualizar o vídeo do cliente
        self.update_client_video()

        # Quando a janela do cliente for fechada, limpar referências
        def on_client_close():
            self.client_window.destroy()
            self.client_window = None
            self.client_video_label = None
            self.client_status_label = None

        self.client_window.protocol("WM_DELETE_WINDOW", on_client_close)

    def update_client_video(self):
        """Atualiza o vídeo na janela do cliente - Clean feed sem textos de OpenCV"""
        if self.client_window is None or not self.client_window.winfo_exists():
            # Janela foi fechada
            self.client_window = None
            self.client_video_label = None
            self.client_status_label = None
            return

        # Usa o mesmo frame e resultado compartilhados
        with self.frame_lock:
            if self.shared_frame is not None:
                frame = self.shared_frame.copy()
                result = self.shared_result
            else:
                frame = None
                result = ""

        if frame is not None:
            # ===== CONVERTER E EXIBIR SEM TEXTOS DE OPENCV =====
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((900, 600), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(image=img)

            self.client_video_label.configure(image=img_tk)
            self.client_video_label.image = img_tk

        # ===== ATUALIZAR STATUS EM TEMPO REAL =====
        if self.client_status_label is not None:
            display_status = ""
            status_color = "gray"

            # Se estiver em modo de cadastro, exibir instrução
            if self.is_enrollment_running and self.current_enrollment_instruction:
                display_status = f"📋 {self.current_enrollment_instruction}"
                status_color = "cyan"
            else:
                # Caso de reconhecimento
                # Limpador rápido para variantes sem acento
                empty_variants = ["", "Aguardando...", "Nao reconhecido", "Não reconhecido"]
                if result in empty_variants:
                    display_status = ""
                    status_color = "gray"
                else:
                    # Caso de spoofing / bloqueio explícito
                    if ("BLOQUEADO" in result) or ("Bloqueado" in result) or ("Spoofing" in result) or ("Acesso bloqueado" in result):
                        display_status = f"❌ {result}"
                        status_color = "red"
                    else:
                        # Extrair a parte principal antes de '|' para uma mensagem curta
                        main = result.split("|")[0].strip()
                        # Se main for um placeholder de não reconhecido, limpar
                        if main in empty_variants:
                            display_status = ""
                            status_color = "gray"
                        else:
                            display_status = f"✅ {main}"
                            status_color = "green"

            self.client_status_label.configure(text=display_status, text_color=status_color)

        # Próxima atualização
        self.after(33, self.update_client_video)  # ~30 FPS

    def update_last_recognized_label(self):
        """Atualiza o texto de 'Última pessoa reconhecida' nas telas"""
        text = f"Última pessoa reconhecida: {self.last_recognized_user}"
        self.last_recognized_label.configure(text=text)
        # Nota: client_status_label agora é atualizado em update_client_video()

    # ------------------------
    # Funções de LOG
    # ------------------------

    def ensure_logs_dir(self):
        """Garante que o diretório de logs exista."""
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
        except Exception as e:
            print(f"Erro ao criar diretório de logs: {e}")

    def get_today_log_path(self):
        """Retorna o caminho do arquivo de log do dia atual."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        return os.path.join(LOG_DIR, f"{date_str}.log")

    def cleanup_old_logs(self):
        """Remove automaticamente logs mais antigos que 6 meses."""
        try:
            if not os.path.exists(LOG_DIR):
                return

            limit_date = datetime.now() - timedelta(days=180)
            for fname in os.listdir(LOG_DIR):
                if not fname.endswith(".log"):
                    continue
                full_path = os.path.join(LOG_DIR, fname)
                date_part = fname[:-4]  # remove .log
                try:
                    file_date = datetime.strptime(date_part, "%Y-%m-%d")
                except ValueError:
                    # Nome não segue o padrão de data, ignora
                    continue

                if file_date < limit_date:
                    try:
                        os.remove(full_path)
                    except Exception as e:
                        print(f"Erro ao remover log antigo '{full_path}': {e}")
        except Exception as e:
            print(f"Erro na limpeza de logs antigos: {e}")

    def load_today_log_to_ui(self):
        """Carrega o log do dia atual na aba de logs."""
        if self.log_textbox is None:
            return

        path = self.get_today_log_path()
        if not os.path.exists(path):
            # Sem arquivo ainda: mensagem padrão
            self.log_textbox.configure(state="normal")
            self.log_textbox.delete("1.0", "end")
            self.log_textbox.insert("1.0", "Sem entradas hoje ainda.")
            self.log_textbox.configure(state="disabled")
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
        except Exception as e:
            content = f"Erro ao ler log de hoje: {e}"

        self.log_textbox.configure(state="normal")
        self.log_textbox.delete("1.0", "end")
        if content:
            self.log_textbox.insert("1.0", content + "\n")
        else:
            self.log_textbox.insert("1.0", "Sem entradas hoje ainda.")
        self.log_textbox.configure(state="disabled")

    def log_recognition(self, user, dist):
        """Registra uma entrada de reconhecimento no log e na aba de logs."""
        try:
            now = datetime.now()
            current_hour = now.hour

            # Verifica se o mesmo usuário e mesma hora já foi registrado
            if self.last_logged_hour == current_hour and self.last_logged_user == user:
                return 
            
            # Atualiza último log
            self.last_logged_hour = current_hour
            self.last_logged_user = user

            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S")
            line = f"[{date_str} {time_str}] {user} (dist={dist:.3f})"

            # Garante diretório e caminho
            self.ensure_logs_dir()
            path = self.get_today_log_path()

            # Escreve no arquivo
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

            # Atualiza UI
            self.append_log_line_to_ui(line)
        except Exception as e:
            print(f"Erro ao registrar log de reconhecimento: {e}")

    def append_log_line_to_ui(self, line):
        """Adiciona uma linha no textbox de log."""
        if self.log_textbox is None:
            return

        self.log_textbox.configure(state="normal")
        current = self.log_textbox.get("1.0", "end").strip()
        if current == "Sem entradas hoje ainda." or current == "":
            self.log_textbox.delete("1.0", "end")
            self.log_textbox.insert("end", line + "\n")
        else:
            self.log_textbox.insert("end", line + "\n")
        self.log_textbox.see("end")
        self.log_textbox.configure(state="disabled")

    def reload_database(self):
        """Recarrega o banco de dados"""
        try:
            self.db = load_database()
            count = len(self.db["labels"]) if self.db["labels"].size > 0 else 0
            self.recognize_status.configure(
                text=f"Banco carregado ({count} embeddings)", text_color="blue"
            )
            self.after(2000, lambda: self.recognize_status.configure(text="Pronto", text_color="gray"))
        except Exception as e:
            self.show_error(f"Erro ao carregar banco: {str(e)}")

    def show_error(self, message):
        """Mostra uma mensagem de erro"""
        # Atualizar status com mensagem de erro
        self.recognize_status.configure(text=f"Erro: {message}", text_color="red")
        print(f"ERRO: {message}")

    def on_closing(self):
        """Callback quando a janela é fechada"""
        self.is_recognition_running = False
        self.is_enrollment_running = False
        if self.cap is not None:
            self.cap.release()
        self.destroy()


if __name__ == "__main__":
    app = FaceRecognitionApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
