# 🔐 Sistema de Reconhecimento Facial com Liveness Detection

Um sistema robusto de reconhecimento e autenticação facial em tempo real com detecção de liveness (anti-spoofing), desenvolvido em Python com interfaces terminal e GUI moderna.

## ✨ Features Principais

- **🎯 Reconhecimento Facial em Tempo Real**: Identificação de usuários através de embeddings faciais (Facenet/DeepFace)
- **🛡️ Detecção de Liveness**: Validação de vida através de:
  - Detecção de piscar de olhos (Eye Aspect Ratio)
  - Detecção de movimento de cabeça
  - Análise de qualidade facial
  - Desafios interativos
- **💾 Cadastro Guiado**: Captura de múltiplas poses faciais (frente, direita, esquerda, cima, baixo, sorrir)
- **⚡ Otimizações de Performance**:
  - Cache global do modelo DeepFace
  - Detecção de movimento para pular frames similares
  - Redimensionamento de frames (320x240)
  - Reconhecimento vetorizado em batch
- **🖥️ Dual UI**:
  - **GUI Moderna**: Interface CustomTkinter com abas, configurações em tempo real
  - **Terminal**: Modo simplificado para linha de comando
- **📊 Sistema de Logs**: Registro automático de entradas com timestamp
- **⚙️ Configurações Ajustáveis**: Thresholds, intervalo de reconhecimento, tamanho de frame, etc.

## 🚀 Início Rápido

### Requisitos

- Python 3.8+
- Câmera (webcam)
- Bibliotecas (veja `requirements.txt`):
  - `deepface==0.0.98` (Modelo Facenet)
  - `opencv-python==4.13.0.90`
  - `customtkinter==5.2.2` (para GUI)
  - `numpy==2.4.1`
  - `pillow`

### Instalação

```bash
# Clone ou extraia o projeto
cd reconhecimento-facial

# Instale as dependências
pip install -r requirements.txt
```

> **Nota**: Na primeira execução, o DeepFace fará download automático do modelo Facenet (~200MB). Pode levar alguns minutos.

### Uso

#### 🖥️ Interface Gráfica (Recomendado)

```bash
python gui.py
```

**Abas disponíveis:**
- **Cadastro**: Registre novos usuários com 6 poses diferentes
- **Reconhecimento**: Reconheça usuários em tempo real com liveness
- **Logs**: Visualize histórico de entradas do dia
- **Configurações**: Ajuste thresholds, FPS, tamanho de frame, etc.

#### 💻 Modo Terminal

```bash
python main.py
```

Escolha:
1. **Cadastro**: Nome do usuário → 6 poses faciais
2. **Reconhecimento**: Reconhecimento em tempo real com estatísticas

## 🏗️ Arquitetura

### Estrutura de Arquivos

```
reconhecimento-facial/
├── gui.py                      # Interface gráfica principal (CustomTkinter)
├── main.py                     # Modo terminal simplificado
├── face_embedding.py           # Geração de embeddings (DeepFace/Facenet)
├── recognize.py                # Reconhecimento com vetorização
├── enroll.py                   # Cadastro guiado de usuários
├── liveness.py                 # Detector de liveness (anti-spoofing)
├── liveness_integration.py     # Integração de liveness ao reconhecimento
├── camera.py                   # Wrapper para captura de câmera
├── benchmark.py                # Ferramenta de benchmark de performance
├── requirements.txt            # Dependências do projeto
├── database/
│   └── users/
│       ├── Lorena/
│       │   └── 0.npy, 1.npy, ... (embeddings)
│       └── Waldemar/
│           └── 0.npy, 1.npy, ...
├── logs/                       # Histórico de entradas (YYYY-MM-DD.log)
└── modelos/
    └── haarcascade_frontalface_default.xml
```

### Fluxo de Dados

#### Cadastro
```
Câmera → Frame (320x240) → Detecção Haar → Embedding (128-D) → Salvar .npy
```

#### Reconhecimento
```
Câmera → Frame (320x240) → Detecção Haar → Validar Liveness
    ↓ (Se vivo)
    Embedding → Comparar com DB (dist. Euclidiana) → Resultado
```

### Módulos Principais

#### `face_embedding.py`
- `get_embedding(frame)` → `np.array` (128-D vector)
- Otimização: Modelo Facenet carregado uma única vez globalmente

#### `recognize.py`
- `load_database()` → Dict com matriz de embeddings + labels
- `recognize(frame, db)` → `(user_name, distance)`
- Operações vetorizadas (Numpy)

#### `liveness.py`
- `LivenessDetector` class com métodos:
  - `detect_eye_blink()` - Detecta piscar
  - `detect_head_movement()` - Detecta movimento
  - `detect_face_quality()` - Valida qualidade
  - `validate_frame()` - Validação simples (1 frame)
  - `validate_liveness()` - Validação robusta (múltiplos frames)

#### `enroll.py`
- `guided_enroll()` - Cadastro terminal
- `guided_enroll_gui_manual()` - Cadastro com captura manual (GUI)

#### `gui.py`
- Interface CustomTkinter com 4 abas
- Threading seguro para captura e reconhecimento
- Configurações dinâmicas em tempo real
- Tela do cliente (segunda janela)
- Sistema de logs persistente

## ⚙️ Configurações

### Principais Parâmetros

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `THRESHOLD` (recognize.py) | 10.0 | Distância Euclidiana máxima para reconhecimento |
| `confidence_threshold` (liveness.py) | 0.7 | Score mínimo de confiança para liveness (0-1) |
| `recognition_interval` (gui.py) | 0.15s | Tempo entre reconhecimentos (reduz processamento) |
| `frame_size` (gui.py) | (240, 320) | Tamanho do frame para processamento |

### Ajustes via GUI

Na aba **Configurações**:
- Threshold de Liveness (0.40-0.95)
- Intervalo de Reconhecimento (0.05-0.50s)
- Distância Máxima (0-∞)
- Detecção de Movimento (ON/OFF)
- Tamanho de Frame (160x120, 320x240, 640x480, Full HD)
- Configurações de Câmera (largura, altura, FPS)

## 📊 Performance & Otimizações

### Otimizações Implementadas

1. **#1 - Cache Global do Modelo**: DeepFace carregado uma única vez
2. **#2 - Detecção de Movimento**: Pula reconhecimento se frames são >98% similares
3. **#3 - Detecção Haar uma única vez**: Reutiliza detecção em cascata
4. **#4 - Redimensionamento de Frame**: 320x240 para velocidade
5. **#5 - LRU Cache de Embeddings**: Memória dos últimos 100 frames

### Benchmarks Esperados

Rode `python benchmark.py` para medir:
- **Latência de Embedding**: ~100-200ms (com cache)
- **FPS de Captura**: ~30 FPS
- **Taxa de Reconhecimento**: ~6-10 reconhecimentos/segundo
- **Uso de Memória**: ~300-500MB (com DB carregado)

```bash
python benchmark.py
# Escolha:
# 1: Latência de Embedding (100 iterações)
# 2: FPS de Captura (30s)
# 3: Performance de Reconhecimento (30s)
# 4: Tudo + Recursos do Sistema
```

## 🔒 Segurança - Liveness Detection

### Como Funciona

O detector valida três critérios:

1. **Piscar de Olhos** (30%): Eye Aspect Ratio varia
2. **Movimento de Cabeça** (30%): Posição muda entre frames
3. **Qualidade da Face** (40%): Tamanho, luminosidade, foco

### Resultado

```python
{
    'is_live': True/False,
    'overall_confidence': 0.85,  # 0-1
    'blink_confidence': 0.75,
    'movement_confidence': 0.92,
    'quality_confidence': 0.88
}
```

### Bloqueio por Spoofing

Se liveness falhar (ex: foto/vídeo):
```
❌ [17:30:45] Acesso bloqueado - Liveness falhou
```

Estatísticas aparecem em tempo real:
- ✅ Passou: 45
- ❌ Bloqueado: 3

## 📝 Uso Prático

### Cadastrar Novo Usuário

**Via GUI:**
1. Abra `gui.py`
2. Vá para aba **Cadastro**
3. Informe o nome (ex: "João Silva")
4. Clique **"Iniciar Cadastro"**
5. Siga as instruções (6 poses)
6. Clique **"Capturar Foto"** para cada pose
7. Sistema salva embeddings em `database/users/João Silva/`

**Via Terminal:**
```bash
python main.py
# Digite: 1 (Cadastro)
# Digite: Nome do usuário
# Pressione Enter para cada pose
```

### Reconhecer Usuários

**Via GUI:**
1. Abra `gui.py`
2. Vá para aba **Reconhecimento**
3. Clique **"Iniciar Reconhecimento"**
4. Apareça na câmera
5. Sistema reconhece automaticamente
6. Abre segunda tela (tela do cliente) com **"Abrir tela do cliente"**

**Via Terminal:**
```bash
python main.py
# Digite: 2 (Reconhecimento)
# Apareça na câmera por 30+ segundos
# Pressione ESC para sair
```

### Visualizar Logs

**Via GUI - Aba "Logs":**
- Mostra todas as entradas do dia com timestamp
- Auto-atualiza

**Via Arquivo:**
```bash
cat logs/2026-01-30.log
# [2026-01-30 17:30:45] João Silva (dist=3.45)
# [2026-01-30 17:35:12] Maria Santos (dist=2.89)
```

## 🎨 Customização

### Mudar Threshold de Reconhecimento

Edite em `recognize.py`:
```python
THRESHOLD = 10.0  # Mais alto = menos rigoroso
```

### Mudar Threshold de Liveness

No GUI → Aba **Configurações** → Slider de Threshold

Ou em código `liveness.py`:
```python
detector = LivenessDetector(confidence_threshold=0.75)
```

### Mudar Intervalo de Reconhecimento

GUI → Aba **Configurações** → Intervalo de Reconhecimento

Ou em `main.py`:
```python
interval = 0.15  # 150ms = ~6.7 FPS
```

## 🐛 Troubleshooting

| Problema | Solução |
|----------|---------|
| Câmera não abre | Verifique permissões, tente usar outro índice em `camera.py` |
| Embedding muito lento | Aumente o intervalo de reconhecimento ou reduza tamanho de frame |
| Liveness bloqueia tudo | Diminua threshold (Configurações → Slider) |
| Não reconhece rosto | Aumente THRESHOLD em `recognize.py`, verifique iluminação |
| Erro "DeepFace model not found" | Deixe fazer download na primeira execução (~200MB, 5 min) |

## 📚 Exemplo de Integração

```python
from recognize import load_database, recognize
from liveness_integration import recognize_with_liveness
from liveness import LivenessDetector
import cv2

# Carregar DB e detector
db = load_database()
detector = LivenessDetector(confidence_threshold=0.7)

# Capturar frame
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

# Detectar rosto
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# Reconhecer COM liveness
if len(faces) > 0:
    result = recognize_with_liveness(frame, faces[0], db, detector)
    
    if result['allowed']:
        print(f"✅ Bem-vindo, {result['user']}!")
    else:
        print(f"❌ {result['reason']}")
```

## 📦 Requisitos do Sistema

- **OS**: Windows, Linux, macOS
- **Python**: 3.8+ (testado em 3.10+)
- **RAM**: Mínimo 2GB (recomendado 4GB+)
- **Câmera**: 640x480 @ 30FPS
- **GPU**: Opcional (accelera embeddings)

## 📄 Licença

Projeto educacional. Sinta-se livre para usar, modificar e distribuir.

## 👨‍💻 Autor

Desenvolvido como sistema de controle de acesso facial com anti-spoofing.

---

## 🔗 Referências

- [DeepFace](https://github.com/serengp/deepface) - Reconhecimento Facial
- [OpenCV](https://opencv.org/) - Visão Computacional
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) - GUI Moderna

## ✅ Checklist de Uso

- [ ] Instalou dependências (`pip install -r requirements.txt`)
- [ ] Testou câmera com `gui.py`
- [ ] Cadastrou pelo menos 1 usuário
- [ ] Testou reconhecimento com liveness
- [ ] Visualizou logs na aba "Logs"
- [ ] Ajustou configurações conforme necessário

---

**Desenvolvido com ❤️ para segurança e reconhecimento facial inteligente.**
