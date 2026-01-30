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
from camera import open_camera

# ===== SEGURANÇA: Liveness Detection =====
from liveness import LivenessDetector
from liveness_integration import recognize_with_liveness

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
        self.liveness_detector = LivenessDetector(confidence_threshold=0.7)
        self.liveness_enabled = True  # Toggle para ativar/desativar
        self.liveness_stats = {
            'checks': 0,
            'passed': 0,
            'failed': 0
        }

        # ===== CONFIGURAÇÕES AJUSTÁVEIS =====
        self.recognition_interval = 0.15  # Intervalo entre reconhecimentos (segundos)
        self.max_distance = None  # Distância máxima para reconhecimento (None = sem limite)
        self.movement_detection_enabled = True  # Detecção de movimento ativada
        self.frame_size = (240, 320)  # Tamanho do frame para processamento (altura, largura)
        
        # ===== CONFIGURAÇÕES DE CÂMERA =====
        self.camera_width = 640  # Largura do frame da câmera
        self.camera_height = 480  # Altura do frame da câmera
        self.camera_fps = 30  # FPS da câmera

        # Info de última pessoa reconhecida
        self.last_recognized_user = "Nenhum"
        self.last_logged_hour = None  # hora do último log
        self.last_logged_user = None  # usuário do último log

        # Variáveis para cadastro manual
        self.capture_event = threading.Event()
        self.current_enroll_frame = None
        self.enroll_frame_lock = threading.Lock()

        # Janela do cliente (segunda tela)
        self.client_window = None
        self.client_video_label = None
        self.client_status_label = None

        # UI de logs
        self.log_textbox = None

        # Preparar diretório de logs e limpeza de arquivos antigos
        self.ensure_logs_dir()
        self.cleanup_old_logs()

        # Criar abas
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)

        # Aba de Cadastro
        self.enroll_tab = self.tabview.add("Cadastro")
        self.setup_enroll_tab()

        # Aba de Reconhecimento
        self.recognize_tab = self.tabview.add("Reconhecimento")
        self.setup_recognize_tab()

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

        # Largura da câmera
        width_container = ctk.CTkFrame(camera_frame, fg_color="transparent")
        width_container.pack(fill="x", padx=15, pady=(10, 5))

        width_label = ctk.CTkLabel(
            width_container,
            text="Largura do Frame:",
            font=ctk.CTkFont(size=11),
        )
        width_label.pack(side="left", padx=(0, 10))

        self.width_var = ctk.StringVar(value=str(self.camera_width))
        self.width_dropdown = ctk.CTkComboBox(
            width_container,
            values=["320", "640", "1280", "1920"],
            variable=self.width_var,
            width=100,
        )
        self.width_dropdown.pack(side="right")

        # Altura da câmera
        height_container = ctk.CTkFrame(camera_frame, fg_color="transparent")
        height_container.pack(fill="x", padx=15, pady=(10, 5))

        height_label = ctk.CTkLabel(
            height_container,
            text="Altura do Frame:",
            font=ctk.CTkFont(size=11),
        )
        height_label.pack(side="left", padx=(0, 10))

        self.height_var = ctk.StringVar(value=str(self.camera_height))
        self.height_dropdown = ctk.CTkComboBox(
            height_container,
            values=["240", "480", "720", "1080"],
            variable=self.height_var,
            width=100,
        )
        self.height_dropdown.pack(side="right")

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

        checks_label = ctk.CTkLabel(
            stats_display_frame,
            text=f"Verificações de Liveness: {self.liveness_stats['checks']}",
            font=ctk.CTkFont(size=10),
        )
        checks_label.pack(side="left", padx=(0, 20))

        passed_label = ctk.CTkLabel(
            stats_display_frame,
            text=f"✅ Passou: {self.liveness_stats['passed']}",
            font=ctk.CTkFont(size=10),
            text_color="#00FF7F",
        )
        passed_label.pack(side="left", padx=(0, 20))

        failed_label = ctk.CTkLabel(
            stats_display_frame,
            text=f"❌ Bloqueado: {self.liveness_stats['failed']}",
            font=ctk.CTkFont(size=10),
            text_color="#FF6B6B",
        )
        failed_label.pack(side="left")

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

    def _set_recognition_interval(self, value):
        """Callback para slider de intervalo de reconhecimento."""
        val = float(value)
        self.recognition_interval = val
        self.interval_display.configure(text=f"{val:.3f}s")

    def _set_max_distance(self, value):
        """Callback para slider de distância máxima."""
        val = float(value)
        if val == 0.0:
            self.max_distance = None
            self.distance_display.configure(text="∞ (ilimitado)")
        else:
            self.max_distance = val
            self.distance_display.configure(text=f"{val:.2f}")

    def _toggle_liveness_config(self):
        """Callback para toggle de liveness na aba de configurações."""
        self.liveness_enabled = self.liveness_toggle.get()

    def _toggle_movement_detection(self):
        """Callback para toggle de detecção de movimento."""
        self.movement_detection_enabled = self.movement_toggle.get()

    def _set_frame_size(self, value):
        """Callback para dropdown de tamanho de frame."""
        size_map = {
            "160x120": (120, 160),
            "320x240": (240, 320),
            "640x480": (480, 640),
            "Full HD": None,  # Sem redimensionamento
        }
        self.frame_size = size_map.get(value, (240, 320))

    def _reset_stats(self):
        """Reseta estatísticas de liveness."""
        self.liveness_stats = {'checks': 0, 'passed': 0, 'failed': 0}
        # Atualizar display (opcional - poderia adicionar labels dinâmicos)

    def _apply_camera_settings(self):
        """Aplica as configurações de câmera - reinicializa a câmera com novos parâmetros."""
        try:
            # Pegar valores dos dropdowns
            new_width = int(self.width_var.get())
            new_height = int(self.height_var.get())
            new_fps = int(self.fps_var.get())
            
            # Atualizar variáveis
            self.camera_width = new_width
            self.camera_height = new_height
            self.camera_fps = new_fps
            
            # Fechar câmera antiga
            if self.cap is not None:
                self.cap.release()
            
            # Reinicializar câmera com novas configurações
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.camera_fps)
            
            # Feedback visual
            self.apply_camera_btn.configure(
                text=f"✅ Aplicado! ({self.camera_width}x{self.camera_height} @ {self.camera_fps}FPS)",
                fg_color="#2d7a2d"
            )
            print(f"✅ Câmera reconfigurada: {self.camera_width}x{self.camera_height} @ {self.camera_fps}FPS")
            
            # Voltar ao texto original após 2 segundos
            self.after(2000, lambda: self.apply_camera_btn.configure(
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
            
            ret, frame = self.cap.read()
            if ret:
                with self.enroll_frame_lock:
                    self.current_enroll_frame = frame.copy()
                
                # Sempre atualizar preview (em tempo real)
                self.after(0, lambda f=frame: self.display_frame_enroll(f))
            
            time.sleep(0.03)  # ~30 FPS

    def update_enroll_progress(self, value):
        """Atualiza a barra de progresso do cadastro"""
        self.after(0, lambda: self.enroll_progress.set(value))

    def update_enroll_status(self, text, color="white"):
        """Atualiza o status do cadastro"""
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

    def capture_thread_func(self):
        """Thread de captura de frames com redimensionamento otimizado"""
        while self.is_recognition_running:
            if self.cap is None:
                time.sleep(0.1)
                continue
            ret, frame = self.cap.read()
            if ret:
                # ===== OTIMIZAÇÃO #4: Redimensionar para 320x240 =====
                frame_resized = cv2.resize(frame, (320, 240))
                with self.frame_lock:
                    self.shared_frame = frame_resized.copy()
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
