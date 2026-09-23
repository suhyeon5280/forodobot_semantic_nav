# Earth Rovers SDK — OmniVLA-edge 자율주행

[frodobots-org/earth-rovers-sdk](https://github.com/frodobots-org/earth-rovers-sdk)
포크. 파인튜닝한 [OmniVLA-edge](https://github.com/NHirose/OmniVLA) 체크포인트로
Earth Rover가 스스로 주행합니다. **웹페이지에 목적지를 글로 적으면 모델이 운전합니다.**

터미널 두 개는 처음에 한 번 띄우고 그 뒤로는 건드리지 않습니다. 조작은 전부 웹에서 합니다.

---

## 바로 실행

**서버와 모델이 같은 컴퓨터에서 터미널 두 개로 돕니다.** conda 환경은 하나면 됩니다.

### 1. 환경 설정 — 최초 1회

```bash
git clone https://github.com/suhyeon5280/forodobot_semantic_nav.git
cd forodobot_semantic_nav

conda env create -f environment.yml -n rover   # 이름은 아무거나. 있으면 다른 이름으로
conda activate rover
python -m playwright install chromium          # 빼먹으면 브라우저가 안 뜹니다
cp .env.sample .env && vi .env                 # SDK_API_TOKEN, BOT_SLUG 채우기
```

**가중치는 깃에 없습니다.** `models/`에 아래 다섯 개를 **이름 그대로** 넣으세요.

| `models/` 안의 이름 | 크기 | 원본 (`E=~/suhyeon/edge_vlm`) |
|---|---|---|
| `arm4p_s0_latest.pth` | 415M | `$E/results/phase4/omnivla/`**`arm4p_s0`**`/`**`latest`**`.pth` |
| `act4_cl6159_s2.pt` | 55M | `$E/results/phase4/d79_ladder/act4_cl6159_s2.pt` |
| `full_H1_linear_lr0.001_s0.pt` | 1.1M | `$E/results/phase4/loc_head_full/full_H1_linear_lr0.001_s0.pt` |
| `yolov8n.pt` | 6.3M | `$E/yolov8n.pt` |
| `omni_deps/` | 12M | `$E/.omni_deps` — **폴더째** 복사, pip 설치 금지 |

> ⚠️ 옆 폴더 `arm1p_s0`는 **3채널**이라 넣으면 `size mismatch` 로 죽습니다. arm4**p**\_s0 의
> **latest**.pth 입니다 — `best.pth` 도 다른 파일입니다.

```bash
sha256sum models/arm4p_s0_latest.pth | cut -c1-16   # 6bc0b5da318d4dd1 이어야 정상
python -m policy.check_ours                         # 로봇 없이 점검. PASS 나와야 함
```

### 2. 실행 — 매번

```bash
# 터미널 1 — SDK 서버  (다른 기기에서 페이지를 열려면 --bind 0.0.0.0:8000)
cd ~/forodobot_semantic_nav && conda activate rover
hypercorn main:app

# 터미널 2 — 모델  (새 터미널, 같은 환경)
cd ~/forodobot_semantic_nav && conda activate rover
python -m policy.run_autonomy --dry-run   # 첫 주행은 반드시 --dry-run
```

브라우저에서 **http://localhost:8000/static/autonomy_control.html** 를 엽니다. 그 뒤
조작은 전부 웹에서 하고 **터미널은 다시 건드리지 않습니다.**

막히면 → [문제 해결](#문제-해결) · [모델 넣기](#2-모델-넣기) · [첫 주행](#첫-주행)

> **이미 clone 해둔 게 있으면** `git pull` 뒤에 `.env`를 손봐야 합니다. `.env`는 레포에
> 없는 파일이라 pull이 고쳐주지 않고, `MISSION_SLUG`가 남아 있으면 모든 엔드포인트가
> 400을 뱉습니다. → [업데이트 받기](#업데이트-받기)

---

## 목차

- [바로 실행](#바로-실행)
- [구조](#구조)
- [준비물](#준비물)
- [최초 1회 설정](#최초-1회-설정)
- [업데이트 받기](#업데이트-받기)
- [실행](#실행)
- [정책 — arm-4′](#정책--arm-4)
- [웹페이지 사용법](#웹페이지-사용법)
- [첫 주행](#첫-주행)
- [⚠️ 주의사항](#️-주의사항)
- [목표 지정 방식](#목표-지정-방식)
- [속도 캘리브레이션](#속도-캘리브레이션)
- [주행 데이터 기록](#주행-데이터-기록)
- [문제 해결](#문제-해결)
- [실행 옵션 전체](#실행-옵션-전체)
- [동작 원리](#동작-원리)
- [파일 구성](#파일-구성)
- [SDK 엔드포인트](#sdk-엔드포인트)
- [출처](#출처)

## 구조

프로세스 두 개가 로봇을 소유한 같은 머신에서 돕니다.

```
                     ┌──────────────────────────────────────┐
  브라우저 ──────────▶│  SDK 서버 (main.py)         :8000    │
  (아무 컴퓨터나)      │    헤드리스 Chrome ─ Agora ─ 로봇     │
        │            └──────────────────────────────────────┘
        │                      ▲                  ▲
        │             GET /v2/front          POST /control
        │             (카메라 프레임)          (linear, angular)
        │                      │                  │
        │            ┌──────────────────────────────────────┐
        └───────────▶│  모델 프로세스               :8010    │
        /state,/cmd  │    models/ ─ arm-4′ 추론              │
                     └──────────────────────────────────────┘
```

**SDK 서버**는 `main.py` 수정 없이 원본 그대로입니다. 헤드리스 브라우저가 로봇의 Agora
채널에 접속하고, 서버는 그걸 HTTP로 열어줍니다.
([browser_service.py](browser_service.py)만 공식 레포를 따라 Playwright로 옮겼습니다 —
아래 [브라우저 드라이버](#브라우저-드라이버-pyppeteer--playwright) 참고.)

**모델 프로세스**는 초당 3번 돕니다: 카메라 프레임을 받아 → 최근 6프레임과 지시문으로
추론 → 예측 궤적을 주행 명령으로 변환 → 전송.

**왜 프로세스를 나눴나:** 모델을 고칠 때마다 재시작하게 되는데, 서버에 합쳐두면 그때마다
로봇의 Agora 연결이 끊깁니다. 미션 모드였다면 주행 기록까지 날아갑니다.

웹페이지는 SDK 서버가 서빙하지만 모델 프로세스와 통신합니다. 그래서 모델이 안 떠 있어도
페이지는 열리고, 어느 쪽이 없는지 화면에 표시합니다.

---

## 준비물

- **Earth Rover**와 [SDK 토큰](https://my.frodobots.com/owner/settings)
- **두 프로세스를 돌릴 머신 1대**
  - NVIDIA GPU (모델이 **CUDA 전용**입니다 — [동작 원리](#동작-원리) 참고)
  - **conda** (Miniconda 또는 Anaconda) — Python은 conda가 깔아줍니다(3.11)
  - 브라우저는 `playwright install chromium`이 알아서 받습니다. Google Chrome이 설치돼
    있으면 그쪽을 먼저 씁니다 — Chrome은 H.264 코덱이 있어서 일부 로봇 스트림이 검게
    나오는 걸 막아줍니다.
- **모델 가중치** — 레포에 없습니다. 깃에 올릴 수 없는 크기라 `models/` 폴더가 **비어
  있는 채로** 들어있고, 거기에 직접 복사해야 합니다. 필수 4개에 약 480 MB,
  대조군과 오프라인 캐시까지 포함하면 약 1.8 GB입니다 → [모델 넣기](#2-모델-넣기)
- **디스크 여유 5 GB 정도** — conda 환경(torch 포함)에 3 GB 남짓, `models/`에 1.8 GB

브라우저는 서버에 접속만 되면 어느 컴퓨터에서 열어도 됩니다.

> **두 프로세스는 반드시 같은 머신에서** 돌아야 합니다.
> [browser_service.py](browser_service.py)가 헤드리스 브라우저를
> `http://127.0.0.1:8000/sdk`로 띄우기 때문에, SDK 서버는 자기 머신의 8000 포트여야
> 합니다.

---

## 최초 1회 설정

**서버와 모델은 같은 컴퓨터 한 대에서 돕니다.** 아래 명령은 전부 그 한 대에서, 전부
**레포 루트에서** 실행합니다. "서버컴/모델컴"이 따로 있는 게 아니라 **같은 컴퓨터의
터미널 두 개**입니다.

먼저 레포를 받습니다. 홈 디렉토리처럼 아무 데나 두면 됩니다.

```bash
cd ~
git clone https://github.com/suhyeon5280/forodobot_semantic_nav.git
cd forodobot_semantic_nav
```

이미 clone 해둔 게 있으면 `cd`만 하고 [업데이트 받기](#업데이트-받기)를 보세요.

여기가 맞는지 확인하세요. 아래 모든 명령의 기준점입니다.

```bash
pwd                    # 지금 위치
ls main.py policy/ environment.yml models/
```

네 개 다 찍히면 제대로 온 겁니다. `No such file or directory`가 뜨면 아직 레포 밖입니다.

> 새 터미널을 열 때마다 `cd`부터 다시 해야 합니다. 코드가 `static/`, `dataset/`,
> `policy/calibration.json`을 **상대 경로**로 쓰기 때문에, 다른 디렉터리에서 실행하면
> 페이지가 404거나 캘리브레이션이 엉뚱한 곳에 저장됩니다.

### 0. conda 설치 확인

```bash
conda --version        # 예: conda 24.7.1
```

버전이 찍히면 이 단계는 건너뛰세요. `command not found`면 Miniconda를 깝니다 (Linux
x86_64 기준):

```bash
curl -L -o ~/miniconda.sh \
  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ~/miniconda.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init bash
```

`conda init` 이후에는 **터미널을 닫았다 새로 열어야** `conda` 명령이 먹습니다
(`source ~/.bashrc`도 됩니다). 새 터미널에서 프롬프트 앞에 `(base)`가 보이면 성공입니다.

```bash
conda --version        # 다시 확인
```

### 1. conda 환경 만들기

**환경은 하나면 됩니다.** 서버와 모델이 같이 들어갑니다. 그리고 **이름은 아무거나 됩니다** —
코드 어디에서도 환경 이름을 읽지 않습니다. `environment.yml`에 `rover`라고 적혀 있는 건
기본값일 뿐입니다.

```bash
conda env create -f environment.yml -n rover     # 이름은 원하는 대로
conda activate rover
```

`-n`이 `environment.yml` 안의 이름을 덮어씁니다. 이미 같은 이름의 환경이 있으면
`conda env create`는 **`prefix already exists`로 멈춥니다.** 그럴 땐 `-n rover2`처럼 다른
이름을 주세요. 기존 환경은 건드리지 않습니다.

프롬프트가 `(rover)`로 바뀝니다. torch, CLIP, EfficientNet을 받느라 몇 GB, 몇 분 걸립니다.

#### `environment.yml` 없이 깔려면

conda 환경을 직접 만들고 싶으면 [requirements-all.txt](requirements-all.txt) 하나로
끝납니다. `environment.yml`과 같은 것을 깔고, 파일 안의 이름에 묶이지 않습니다.

```bash
conda create -n rover python=3.11 git -c conda-forge -y
conda activate rover
pip install -r requirements-all.txt
```

#### requirements 파일이 왜 세 개인가

| 파일 | 무엇 | 누가 또 쓰나 |
|---|---|---|
| [requirements.txt](requirements.txt) | SDK 서버 | [Dockerfile](Dockerfile)이 이것만 깝니다. 공식 레포와 동일하게 유지 |
| [policy/requirements.txt](policy/requirements.txt) | 모델 프로세스 | `models/omni_deps`를 갈아끼운 뒤 이것만 따로 돌립니다 |
| [requirements-all.txt](requirements-all.txt) | 위 둘 + `upload_to_hf.py` | 새 환경에 한 번에 깔 때 |

**기존 환경에 얹지 마세요.** 루트 `requirements.txt`는 공식 레포에 맞추느라
`numpy==1.26.4`와 `opencv-python-headless==4.9.0.80`이 **고정**돼 있습니다. 이미 torch가
깔린 다른 환경에 얹으면 numpy가 다운그레이드되면서 그 환경의 torch가 깨집니다. 새로
파는 게 맞습니다. (그 둘은 이 포크의 서버 코드가 import 하지도 않습니다 — 공식
레포에서 그대로 물려받은 것입니다.)

**pip으로 안 깔리는 게 두 개** 있습니다. 둘 다 일부러 그렇게 둔 것입니다.

- `models/omni_deps` (ultralytics, open_clip) — `--no-deps`로 깔린 것이라 폴더째 복사합니다.
  이유는 [아래](#omni_deps는-pip로-설치하지-마세요). 넣는 방법은 [2. 모델 넣기](#2-모델-넣기).
- Chromium — 패키지가 아니라 브라우저입니다. 바로 아래 줄.

그 다음 **헤드리스 브라우저를 받습니다. 이 줄을 빼먹으면 서버가 브라우저를 못 띄웁니다:**

```bash
python -m playwright install chromium
```

Google Chrome이 이미 깔려 있으면 서버가 그쪽을 먼저 씁니다(H.264 코덱 때문에 영상이 더
잘 나옵니다). 특정 브라우저를 강제하려면 `.env`의 `CHROME_EXECUTABLE_PATH`를 쓰세요 —
**설정 안 해도 됩니다.**

잘 만들어졌는지, **GPU가 실제로 잡히는지**까지 확인하세요:

```bash
python --version       # Python 3.11.x
which hypercorn        # .../envs/rover/bin/hypercorn
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`True`가 나와야 합니다. `False`면 여기서 멈추고 드라이버/torch부터 해결하세요 — **모델은
CUDA 전용이라 CPU로는 아예 못 돕니다.** torch는 일부러 conda 패키지가 아니라 **pip으로,
버전 고정 없이** 받습니다. RTX 50 시리즈(Blackwell)는 최신 CUDA 빌드가 필요한데
conda-forge의 pytorch는 한 발 늦어서, 구버전이 깔리면 첫 추론에서 `no kernel image`
에러가 납니다.

환경을 고쳐 만들려면 (`rover` 자리에 실제 쓴 이름을 넣으세요):

```bash
conda env update -f environment.yml -n rover --prune   # yml 변경분만 반영
pip install -r requirements-all.txt                    # requirements만 바뀌었을 때
conda env remove -n rover                              # 통째로 지우고 다시 create
```

### 2. 모델 넣기

가중치는 깃에 없습니다. GitHub는 파일당 100 MB가 한도인데 정책 하나가 415 MB입니다.
그래서 **`models/` 폴더가 빈 상태로 커밋돼 있고**, 거기에 파일을 복사하면 됩니다.
폴더를 새로 만들 필요도, 경로를 설정할 필요도 없습니다.

clone 직후 상태를 먼저 확인하세요.

```bash
ls models/
# README.md    ← 이것만 보이면 정상. 비어 있는 게 맞습니다.
```

#### 이미 `models/`가 있는 컴퓨터에서 옮기는 경우

**보통 이 경우입니다.** 골라 담을 필요 없이 **`models/` 안의 내용을 전부 복사**해서
새 컴퓨터의 `models/` 안에 그대로 넣으세요. USB로 옮기든 `scp`로 보내든 같습니다.
전부 1.8 GB입니다.

폴더 이름과 파일 이름은 **바꾸지 마세요.** 코드가 그 이름으로 찾습니다.

```bash
# 보내는 쪽 — 뽑기 전에 sync 를 꼭 하세요. 파일 관리자가 끝났다고 해도
# 버퍼에 남아 있을 수 있고, 415 MB 짜리가 잘리면 로드에서 죽습니다.
cd ~/forodobot_semantic_nav/models
sha256sum *.pth *.pt > SHA256SUMS          # 받는 쪽에서 대조할 목록
cp -r . /media/$USER/<USB이름>/models/
sync
```

```bash
# 받는 쪽
cd ~/forodobot_semantic_nav
cp -r /media/$USER/<USB이름>/models/. models/
cd models && sha256sum -c SHA256SUMS       # OK 가 다섯 줄 나와야 정상
```

`scp`로 바로 보낼 수도 있습니다.

```bash
scp -r ~/forodobot_semantic_nav/models/. <노트북>:~/forodobot_semantic_nav/models/
```

해시 대조가 귀찮으면 건너뛰고 [오프라인 점검](#4-오프라인-점검)만 돌리세요. 파일이
깨졌으면 거기서 걸립니다.

> **USB에서 `Filesystem does not support symbolic links` 가 뜨면** — `hf/` 안에
> 심볼릭 링크가 남아 있는 것입니다. Hugging Face 캐시는 같은 파일을 중복 저장하지
> 않으려고 `blobs/`에 실제 파일을 두고 `snapshots/`에서 링크로 가리킵니다. USB가
> FAT32/exFAT이면 링크를 저장할 수 없어 복사가 거기서 멈춥니다.
> 이 레포의 `models/hf`는 링크를 이미 풀어두었지만, 캐시를 새로 만들면 다시 생깁니다.
> 그럴 때는 이렇게 푸세요.
>
> ```bash
> cd ~/forodobot_semantic_nav/models
> cp -rL hf hf_flat && rm -rf hf_flat/hub/*/blobs   # -L 이 링크를 실제 파일로 풉니다
> rm -rf hf && mv hf_flat hf
> find models -type l | wc -l                       # 0 이어야 합니다
> ```

**인터넷이 연결되는 노트북이면** `clip/`과 `hf/`는 빼도 됩니다. 첫 실행에 알아서
받습니다. 그러면 옮길 양이 1.8 GB에서 **906 MB로 줄어듭니다.** 다만 그 두 폴더를
가져가면 첫 실행이 빠르고 현장에서 네트워크가 끊겨도 돕니다. 용량이 문제가 아니면
**그냥 전부 복사하는 쪽이 낫습니다.**

#### 폴더에 뭐가 들어있는 건지

| 파일 | 크기 | 역할 | |
|---|---|---|---|
| `arm4p_s0_latest.pth` | 415M | 정책 (arm-4′) | 필수 |
| `act4_cl6159_s2.pt` | 55M | CLIP 어댑터 | 필수 |
| `full_H1_linear_lr0.001_s0.pt` | 1.1M | 국소화 헤드 | 필수 |
| `yolov8n.pt` | 6.3M | 검출기 | 필수 |
| `omni_deps/` | 12M | ultralytics, open_clip | 필수 |
| `arm1_latest.pth` | 418M | 대조군 정책 (`--arm1`) | arm-4′만 쓰면 **불필요** |
| `frames/` | 0.5M | 점검용 프레임 | 점검에 필요 |
| `clip/` | 338M | CLIP ViT-B/32 캐시 | 인터넷 있으면 생략 가능 |
| `hf/` | 571M | CLIP ViT-B/16 캐시 | 인터넷 있으면 생략 가능 |

#### `models/`를 처음 만드는 경우

어느 컴퓨터에도 `models/`가 없다면 학습 저장소에서 모아야 합니다. **한 번만** 하면
되고, 그 뒤로는 위처럼 폴더째 복사하면 됩니다.

원본 경로는 이렇습니다. 이름을 바꿔 복사하세요.

```bash
E=~/suhyeon/edge_vlm
O=~/suhyeon/OmniVLA_edge

cp $E/results/phase4/omnivla/arm4p_s0/latest.pth            models/arm4p_s0_latest.pth
cp $E/results/phase4/d79_ladder/act4_cl6159_s2.pt           models/
cp $E/results/phase4/loc_head_full/full_H1_linear_lr0.001_s0.pt  models/
cp $E/yolov8n.pt                                            models/
cp -r $E/.omni_deps                                         models/omni_deps

# 선택: 대조군
cp $O/train/logs_frodo_lan_ft_full_lang/*/latest.pth         models/arm1_latest.pth

# 선택: 점검용 프레임 몇 장
mkdir -p models/frames/episode_0020/image
(cd $O/omnivla_dataset_hf/episode_0020/image && ls *.jpg | sort | head -16 \
   | xargs -I{} cp {} ~/forodobot_semantic_nav/models/frames/episode_0020/image/)

# 권장: CLIP 캐시. 있으면 첫 실행이 빠르고 네트워크 없이도 돕니다
mkdir -p models/clip models/hf/hub
cp ~/.cache/clip/ViT-B-32.pt                                models/clip/
cp -r ~/.cache/huggingface/hub/models--timm--vit_base_patch16_clip_224.openai \
      models/hf/hub/
```

다 모았으면 해시 목록을 만들어 두세요. 다음부터 다른 컴퓨터로 옮길 때 이 파일로
대조합니다.

```bash
(cd models && sha256sum *.pth *.pt > SHA256SUMS)
```

#### 제대로 들어갔는지 확인

```bash
sha256sum models/*.pth models/*.pt | cut -c1-16,65-
```

| 앞 16자리 | 파일 |
|---|---|
| `6bc0b5da318d4dd1` | `arm4p_s0_latest.pth` |
| `f64da745b6f7214b` | `arm1_latest.pth` |
| `ec1c47b50855a3a7` | `act4_cl6159_s2.pt` |
| `9bc66758f1ac6e69` | `full_H1_linear_lr0.001_s0.pt` |
| `f59b3d833e2ff32e` | `yolov8n.pt` |

#### `omni_deps`는 pip로 설치하지 마세요

`ultralytics`와 `open_clip`은 **`--no-deps`로 설치된 것**이라 디렉토리째 복사해야
합니다. 그냥 `pip install ultralytics`를 하면 자기가 의존하는 torch를 끌고 와서
지금 설치된 torch를 덮어씁니다. 그러면 GPU 커널이 안 맞아 첫 추론에서 죽습니다.

`--no-deps`로 깔렸으니 그 둘의 의존 패키지는 같이 오지 않습니다. 실제 추론에서
쓰이는 것들(`matplotlib`, `lmdb`, `safetensors`, `huggingface-hub`)은
[policy/requirements.txt](policy/requirements.txt)에 적어뒀으므로 `rover` 환경을
만들 때 같이 깔립니다. 예전에 만든 환경이 있으면 한 번 갱신하세요.

```bash
conda activate rover
pip install -r policy/requirements.txt
```

#### `clip/`과 `hf/`는 인터넷이 있으면 생략

CLIP 백본 두 개는 첫 실행 때 자동으로 받아서 `~/.cache`에 넣습니다. 폴더를 복사해
두면 그쪽을 먼저 쓰고, **네트워크를 아예 안 탑니다.** 현장에서 인터넷이 없거나 느릴
거면 미리 복사하세요.

### 3. `.env` 만들기

이 파일이 없으면 **서버가 인증 단계에서 죽습니다.** 레포 루트에 `.env`라는 이름으로
만드세요.

```bash
cp .env.sample .env
```

**`MISSION_SLUG`는 자유 주행에 필요 없습니다 — 공식 SDK에서도 선택사항입니다.** 문제는
"없어도 된다"가 아니라 **"있으면 안 된다"** 는 겁니다. 값이 들어있으면 서버가 미션 모드로
들어가서, `/start-mission`을 호출하기 전까지 `/`를 포함한 모든 엔드포인트가 400
`Call /start-mission endpoint to start a mission`을 뱉습니다.

지금 `.env.sample`에는 주석 처리돼 있지만, **예전에 복사해둔 `.env`가 있다면 직접
확인하세요:**

```bash
grep -n MISSION_SLUG .env
```

최종적으로 이 네 줄이면 됩니다:

```bash
SDK_API_TOKEN="발급받은_토큰"
BOT_SLUG="봇_슬러그"
IMAGE_FORMAT=jpeg
IMAGE_QUALITY=0.8
```

`CHROME_EXECUTABLE_PATH`는 **이제 안 넣어도 됩니다.** 서버가 알아서 Google Chrome →
Playwright 번들 Chromium 순으로 찾습니다. 특정 브라우저를 강제할 때만 쓰세요.

### 4. 오프라인 점검

**로봇에 붙이기 전에 반드시 여기서 통과시키세요.** 서버도 로봇도 필요 없습니다.
`(rover)` 환경에서 레포 루트에 있으면 바로 실행됩니다.

```bash
python -m policy.check_ours
```

세 가지를 봅니다.

1. **regression** — `--upstream` 경로(`best.pth`)가 여전히 도는지. `best.pth`는 릴리스
   첨부 파일이라 clone에는 없습니다. **없으면 `SKIP`이고 실패가 아닙니다.**
2. **port** — 지정 프레임에서 어순을 바꾼 두 문장이 **오프라인 참조와 ±0.01 m 안에서**
   맞는지. 전처리가 한 군데라도 틀어지면 여기서 잡힙니다. 같은 단계에서 후보 점수가
   참조의 `crop_cos` 와 소수점 오차 안에서 같은지도 대조합니다.
3. **timing** — tick 시간이 333 ms 예산 안인지.

정상이면 마지막에 이렇게 나옵니다.

```
  "the white van next to the red truck"
    endpoint lateral: ours -0.20499 m   reference -0.20499 m   delta 0.00000 m   PASS
  "the red truck next to the white van"
    endpoint lateral: ours +0.15277 m   reference +0.15277 m   delta 0.00000 m   PASS
  PASS: port reproduces the reference
  ...
  === summary ===
    regression  SKIP     ← best.pth 가 없으면 정상
    port        PASS
    timing      PASS
```

`models/`에 파일이 빠져 있으면 **무엇이 없는지 목록으로** 알려줍니다.

`models/frames/`를 복사하지 않았다면 port와 timing이 프레임을 못 찾습니다. 전체
데이터셋이 있는 컴퓨터라면 그쪽을 가리키세요.

```bash
OMNIVLA_DATASET_ROOT=~/suhyeon/OmniVLA_edge/omnivla_dataset_hf \
  python -m policy.check_ours
```

대조군까지 같이 점검하려면 `--with-arm1`을 붙입니다 (`arm1_latest.pth` 필요).

```bash
python -m policy.check_ours --with-arm1
```

**port가 FAIL이면 거기서 멈추세요.** 가중치나 전처리가 어긋난 상태라 주행 결과를
해석할 수 없습니다.

여기까지 끝나면 최초 설정은 끝입니다. 이후로는 [실행](#실행)의 터미널 두 개만 반복합니다.

---

## 업데이트 받기

로봇을 돌리는 컴퓨터가 개발하는 컴퓨터와 다르다면, 이미 `git clone` 해둔 쪽에서는 이렇게
갱신합니다. **`git pull`만으로는 부족합니다** — 의존성이 바뀌었으면 환경도 같이 갱신해야
합니다.

```bash
cd ~/forodobot_semantic_nav
git pull

conda activate rover
conda env update -f environment.yml -n rover --prune   # requirements가 바뀌었을 때 (이름은 실제 쓰는 것으로)
python -m playwright install chromium           # 브라우저가 아직 없다면
```

`git pull`이 로컬 변경 때문에 막히면, 그 컴퓨터에서 고친 게 없는 경우엔 이걸로 덮어씁니다:

```bash
git fetch origin && git reset --hard origin/main
```

`best.pth`와 `dataset/`은 gitignore라 pull이 건드리지 않습니다. 그대로 남습니다.

### ⚠️ `.env`는 pull로 안 고쳐집니다

레포에 `.env` 파일 자체가 없기 때문에, **`git pull`은 여러분의 `.env`를 만들지도 고치지도
않습니다.** 아래 두 가지는 노트북에서 직접 해야 합니다.

**1. `MISSION_SLUG` 지우기 (예전에 `cp .env.sample .env` 한 클론이면 100% 해당)**

`.env.sample`에 예제 값이 들어있어서 그대로 복사됐습니다. 이게 들어있으면 **서버의 모든
엔드포인트가 400을 뱉습니다** — `/`에 들어가도 페이지 대신
`{"detail":"Call /start-mission endpoint to start a mission"}` JSON만 보입니다.

```bash
grep -n MISSION_SLUG .env                          # 있는지 확인
sed -i 's/^MISSION_SLUG=/# MISSION_SLUG=/' .env    # 주석 처리
```

**고친 뒤 서버를 반드시 재시작하세요.** `.env`는 프로세스가 뜰 때 한 번만 읽습니다.

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/     # 200이면 통과
```

**2. `CHROME_EXECUTABLE_PATH` 지우기**

예전 `.env.sample`에는 `"/path/to/chrome"`이라는 **플레이스홀더가 실제 값으로** 들어있었습니다.
그대로 두면 `Failed to launch chromium because executable doesn't exist at /path/to/chrome`이
납니다.

```bash
sed -i 's/^CHROME_EXECUTABLE_PATH=/# CHROME_EXECUTABLE_PATH=/' .env
```

이제 이 변수는 선택사항입니다. 없으면 설치된 Google Chrome → Playwright 번들 Chromium
순으로 자동 탐색합니다. (존재하지 않는 경로가 들어와도 무시하고 자동 탐색으로 넘어가지만,
`.env`는 정리해두는 게 낫습니다.)

> **pyppeteer 시절에 받아둔 클론이라면** — 서버가 Playwright로 바뀌었으니 환경을 새로 만드는
> 게 깔끔합니다. 예전 `rover-sdk`/`rover-policy` 환경은 지워도 됩니다.
>
> ```bash
> conda env remove -n rover-sdk
> conda env remove -n rover-policy
> conda env create -f environment.yml -n rover
> conda activate rover
> python -m playwright install chromium
> ```

---

## 실행

**같은 컴퓨터에서 터미널 두 개**를 띄웁니다. 서버컴과 모델컴이 따로 있는 게 아니라, 한
대에서 프로세스 두 개가 도는 구조입니다. 두 터미널 모두 매번 이 세 가지를 순서대로 합니다:

1. **`cd` 레포 루트** — 상대 경로를 쓰기 때문에 필수
2. **`conda activate rover`** — 두 터미널 **같은 환경**입니다
3. 프로세스 실행

### 터미널 1 — SDK 서버

```bash
cd ~/forodobot_semantic_nav
conda activate rover
hypercorn main:app
```

프롬프트가 `(rover)`로 바뀐 걸 확인하고 실행하세요. `Running on http://127.0.0.1:8000`이
뜨면 서버가 산 겁니다. 이 터미널은 그대로 두세요 — 닫으면 서버가 죽습니다.

#### `--bind`는 언제 필요한가

`--bind`는 **접속할 주소가 아니라, 서버가 어느 네트워크 인터페이스에서 들을지**를 정하는
값입니다. hypercorn의 기본값이 이미 `127.0.0.1:8000`이라 **한 대에서 다 돌리면 아무것도 안
붙이면 됩니다.**

| 상황 | 명령 |
|---|---|
| 서버·모델·브라우저 **전부 같은 컴퓨터** | `hypercorn main:app` (기본값 = `127.0.0.1:8000`) |
| 폰·다른 노트북에서 페이지를 열고 싶다 | `hypercorn main:app --bind 0.0.0.0:8000` |

`0.0.0.0`은 "모든 인터페이스에서 듣겠다"는 뜻이지 접속 주소가 아닙니다. 그렇게 띄워도
같은 컴퓨터에서는 여전히 `http://localhost:8000`으로 들어갑니다. 다만 같은 네트워크의
아무나 접속할 수 있게 되니, 혼자 쓸 거면 기본값이 낫습니다.

기본값으로 띄워도 나머지는 그대로 동작합니다 — 헤드리스 브라우저는
`http://127.0.0.1:8000/sdk`로, 모델 프로세스는 `http://localhost:8000`으로 붙는데 둘 다
같은 머신의 루프백이기 때문입니다.

> **hypercorn을 꼭 써야 하나?** 아닙니다. `main.py`는 평범한 FastAPI(ASGI) 앱이라
> WebSocket도 HTTP/2도 안 씁니다. uvicorn도 그대로 됩니다: `uvicorn main:app`.
> hypercorn을 기본으로 두는 건 공식 레포가 그렇게 하고 `requirements.txt`에 이미
> 들어있기 때문입니다. `hypercorn: command not found`가 났다면 hypercorn 잘못이 아니라
> **환경을 activate 안 한 것**입니다.

서버가 떴는지 확인은 **세 번째 터미널**에서 (서버 터미널은 로그가 흐르고 있어서 명령을 못
칩니다):

```bash
curl -s http://127.0.0.1:8000/data | head -c 300
```

### 터미널 2 — 모델

**새 터미널을 열고** — 터미널 1은 서버가 점유 중입니다:

```bash
cd ~/forodobot_semantic_nav
conda activate rover
python -m policy.run_autonomy
```

체크포인트 경로를 줄 필요가 없습니다. `models/`에서 알아서 찾습니다.

프롬프트가 `(rover)`인지 확인하세요. `(base)`인 채로 실행하면
`ModuleNotFoundError: No module named 'torch'`가 납니다 — activate를 빼먹은 겁니다.

서버의 헤드리스 브라우저가 로봇의 Agora 채널에 붙어 첫 프레임을 뱉을 때까지 최대 1분
기다립니다. `operator UI: ...` 로그가 뜨면 준비 완료입니다.

**첫 주행이라면 여기서 `--dry-run`을 붙이세요** → [첫 주행](#첫-주행)

이 명령은 파인튜닝한 **arm-4′**를 돌립니다 → [정책 — arm-4′](#정책--arm-4)

### activate 없이 한 줄로 (선택)

`conda run`을 쓰되 **`--no-capture-output`을 꼭 붙이세요.** 없으면 출력이 버퍼링돼서
`operator UI: ...` 같은 로그가 실시간으로 안 보입니다.

```bash
cd ~/forodobot_semantic_nav
conda run --no-capture-output -n rover hypercorn main:app                    # 터미널 1
conda run --no-capture-output -n rover python -m policy.run_autonomy        # 터미널 2
```

### 브라우저

같은 컴퓨터에서 열면:

```
http://localhost:8000/static/autonomy_control.html
```

다른 컴퓨터(노트북, 태블릿)에서 열면 — 서버 컴퓨터의 IP를 먼저 확인하고:

```bash
hostname -I | awk '{print $1}'      # 예: 192.168.0.12
```

```
http://192.168.0.12:8000/static/autonomy_control.html
```

헤더에 초록색 `policy connected` 배지가 뜨면 양쪽 다 살아있는 겁니다. 이후로는 터미널을
볼 일이 없습니다. 조작은 전부 웹에서 합니다.

### 종료

각 터미널에서 **Ctrl+C** 한 번씩. 모델 프로세스는 종료 경로마다 로봇에 정지 명령을 보내고
나갑니다. 순서는 상관없지만, **모델(터미널 2)을 먼저 내리는 쪽이 안전합니다** — 서버를
먼저 죽이면 모델이 `/control`을 못 보내는 상태가 잠깐 생깁니다.

conda 환경에서 빠져나오려면 `conda deactivate`. 환경을 지울 필요는 없습니다 — 다음에
켤 때 [실행](#실행)의 세 줄만 다시 치면 됩니다.

### 다시 켤 때 (요약)

설정을 한 번 끝냈다면 매번 이것만 하면 됩니다.

```bash
# 터미널 1
cd ~/forodobot_semantic_nav && conda activate rover
hypercorn main:app

# 터미널 2
cd ~/forodobot_semantic_nav && conda activate rover
python -m policy.run_autonomy

# 브라우저: http://localhost:8000/static/autonomy_control.html
```

---

## 정책 — arm-4′

`run_autonomy`는 파인튜닝한 **arm-4′** 정책을 돌립니다. 체크포인트는
`models/arm4p_s0_latest.pth`이고, 플래그 없이 기본으로 선택됩니다.

```bash
conda activate rover
python -m policy.run_autonomy
```

모든 경로가 **레포 안에서만** 해결됩니다. 정책 코드와 설정은
[policy/refs/](policy/refs/)에 커밋돼 있고, 가중치는 `models/`에서 읽습니다.
`ultralytics`와 `open_clip`도 `models/omni_deps`에서 읽으므로 `PYTHONPATH`를 줄
필요가 없습니다. 다른 곳에 두었다면 `MODELS_DIR`, `OMNIVLA_REFS_ROOT`, `OMNI_DEPS`
환경변수로 바꿀 수 있습니다.

`policy/refs/`는 `edge_vlm`과 `OmniVLA_edge`에서 **바이트 단위로 그대로** 복사한
것이고, 손대지 않습니다. 원본 디렉토리 구조까지 맞춰 둔 이유도 같습니다. 그 모듈
몇 개가 import 시점에 상대 경로로 데이터 파일을 읽기 때문에, 구조를 유지하는 것이
사본을 고치지 않는 유일한 방법입니다. 사본을 고치기 시작하면 원본과 조용히 어긋납니다.

> 코드에는 대조군 `--arm1`(파인튜닝 전 원본)과 `--upstream`(레포에 원래 있던
> OmniVLA-edge)도 남아 있습니다. 현장 A/B가 필요해지면 `--help`를 보세요. 둘 다
> **3채널** 체크포인트를 쓰므로 arm-4′ 자리에 넣으면 `size mismatch`가 납니다.

### arm-4′가 매 tick 하는 일

정책 forward와 웨이포인트→제어는 **원본 그대로**입니다. 달라지는 것은 정책을 부르기
직전의 입력 준비뿐입니다.

1. 프롬프트를 `"A next to|beside|near B"`로 파싱합니다. 관계어가 없으면 A가 문장
   전체, B는 없습니다.
2. YOLOv8n(conf 0.25)으로 현재 프레임의 후보 박스를 뽑고, A의 머리명사가 COCO
   클래스로 매핑되면 그 클래스만 남깁니다. 매핑이 안 되거나 해당 클래스가 0개면 전
   박스를 씁니다.
3. 후보 박스를 10% 넓혀 **원본 해상도에서 잘라내고**, CLIP ViT-B/16 + 어댑터로
   임베딩해 A 문구와 코사인 유사도를 냅니다. 국소화 헤드는 지나지 않습니다.
   헤드는 명사를 찾을 뿐 **형용사를 읽지 않아서**, 색만 다른 후보를 구분하지
   못합니다. 이전 구현은 헤드 heatmap의 박스 내 최대값을 점수로 썼고, 그래서
   "black chair"와 "orange chair"가 같은 의자를 골랐습니다. 그 값은 지금도
   `heat_max`로 로그에 같이 남습니다.
4. B도 **3번과 같은 방식**으로 찾습니다. 검출 박스를 각각 잘라 B 문구와 코사인
   유사도를 내고, 가장 높은 박스의 **중심**을 기준 위치로 씁니다. 박스는 B 자신의
   COCO 클래스로 걸러냅니다 — "the orange chair next to the table"이면 table
   박스에서 찾습니다. A와 같은 클래스면 3번에서 만든 crop 벡터를 그대로
   쓰므로 forward가 늘지 않습니다. B가 없으면 건너뜁니다.

   B가 COCO 밖이면(나무, 문, 표지판) 박스가 없으므로 **예전 방식인 heatmap
   피크로 돌아갑니다.** 그 heatmap은 국소화 헤드를 지나서 명사만 보고 색을 못
   읽습니다. 의자가 여럿이면 아무 의자에나 꽂힐 수 있어서, tick 로그에
   `B_mode: "heatmap_peak"` 와 `fallback: "anchor_not_boxed"` 로 남습니다.
5. **B 박스와 IoU 0.5를 넘게 겹치는 A 후보를 뺀 다음**,
   `argmax_k [점수_k − 0.25 · 기준까지의 거리]`로 하나를 고릅니다. B를 빼지 않으면
   B 자신이 A 후보로 남아 **자기와의 거리 0**이라 규칙이 줄 수 있는 최대 가점을
   가져갑니다. B가 없으면 점수만 봅니다.
6. 선택한 박스 안만 남긴 heatmap을 224×224 1채널로 만듭니다.
7. **`current_img`의 RGB 3채널을 0으로 채우고** heatmap을 4번째 채널로 붙입니다.
   정규화 공간의 0은 데이터셋 평균입니다. 이 마스킹이 빠지면 정책이 heatmap을 무시합니다.
   히스토리 6프레임(`obs_img`)은 건드리지 않고, 텍스트 인코더도 원본 CLIP ViT-B/32
   그대로입니다.

후보가 0개면 A heatmap 피크에 고정 크기 박스 하나를 놓고 계속 진행합니다. 참조
구현은 이 경우 arm-1로 넘어가지만, 현장 시험 중에 모델이 조용히 바뀌지 않도록
배포 쪽은 그렇게 하지 않습니다. 박스 크기(한 변 0.25)는 참조에 없던 값이라 여기서
정한 것입니다.

### 로봇에 붙이기 전 점검

```bash
conda activate rover
python -m policy.check_ours --with-arm1
```

절차와 정상 출력은 [4. 오프라인 점검](#4-오프라인-점검)에 있습니다. 실측 지연은
[지연](#지연)에 있습니다.

기본으로 tick 로그가 `field_log/`에 쌓입니다. tick마다 프롬프트·파싱·후보·점수·
선택·웨이포인트·단계별 시간이 `ticks.jsonl`로, 정책 입력 heatmap이 `thumbs/*.png`로
들어갑니다. 위치를 바꾸려면 `--tick-log <디렉토리>`를 주세요.

### 한 장으로 보기

사진 하나를 넣으면 검출 → heatmap → 선택 → 정책 입력 채널 → 궤적까지 한 장에
그립니다. 로봇도 서버도 필요 없습니다.

```bash
python -m policy.visualize_pipeline test.jpg \
  --prompt "the chair next to the monitor" --out viz/out.png
```

프롬프트가 왜 그렇게 해석됐는지, 어느 박스가 왜 뽑혔는지(점수와 거리 항까지),
정책이 실제로 받은 4번째 채널이 어떻게 생겼는지가 그림과 숫자로 같이 나옵니다.
현장 명령을 정하기 전에 후보 문장을 여기에 넣어 보면 됩니다.

`--compare`를 주면 점수 경로 두 가지(`heat_max`와 `crop_cos`)가 같은 프레임에서
각각 어느 박스를 고르는지 나란히 그립니다.

`--anchor-compare`는 **기준 물체(B) 를 찾는 두 방식**을 나란히 놓습니다. `--prompt`를
여러 번 주면 프롬프트당 한 줄씩 그려서 **어순을 바꾼 문장을 같은 장에서** 비교할 수
있습니다. 현장 나가기 전 로버 카메라로 찍은 프레임으로 한 번 돌려보기를 권합니다.

```bash
python -m policy.visualize_pipeline test.jpg --anchor-compare \
  --prompt "the orange chair next to the black chair" \
  --prompt "the black chair next to the orange chair" \
  --out viz/anchor.png
```

### 눈금이 두 개입니다

참조 쪽 지표와 배포 쪽 주행은 **다른 눈금**을 씁니다. 같은 `(8,4)` 출력을 놓고도
숫자가 달라지므로 결과를 나란히 놓을 때 반드시 병기하세요.

| | 참조 지표 | 배포 주행 |
|---|---|---|
| 웨이포인트 간격 | 0.125 m | `control.py`의 `METRIC_WAYPOINT_SPACING = 0.1` |
| 보는 지점 | 끝점(index 7) | `WAYPOINT_INDEX = 4` |

[policy/check_ours.py](policy/check_ours.py)는 두 눈금을 모두 출력합니다.

---

## 웹페이지 사용법

| 조작 | 동작 |
|---|---|
| 지시문 입력 → **Start** | 적은 곳을 향해 주행 시작 |
| 지시문 수정 → **Enter** / **Set instruction** | **정지 없이** 지시문 교체 |
| **Stop** | 루프 일시정지 + 로봇 정지 |
| **E-STOP** / **Space** | 즉시 정지 (모델 프로세스를 거치지 않음) |
| **Start Recording** | `dataset/sessions/`에 주행 기록 |
| **Drive straight 10s** / **Spin 10s** | [속도 캘리브레이션](#속도-캘리브레이션)용 |

화면에는 실시간 카메라, 모델이 예측한 궤적(위에서 본 그림, 로봇이 아래), 지금 전송 중인
명령, 추론 시간이 표시됩니다.

배지 두 개를 보세요:

- **`policy connected` / `policy offline`** — 모델 프로세스가 살아있는지. offline이면
  E-STOP 말고 전부 비활성화됩니다.
- **`RUNNING` / `PAUSED` / `UNKNOWN`** — 로봇 상태. 모델이 죽으면 `UNKNOWN`입니다.
  상태를 알 수 없는데 `RUNNING`이라고 표시하면 안 되니까요.

---

## 첫 주행

**반드시 `--dry-run`으로 시작하세요.** 터미널 1(서버)은 그대로 두고, 터미널 2의 모델만
`--dry-run`으로 다시 띄웁니다.

```bash
cd ~/forodobot_semantic_nav
conda activate rover
python -m policy.run_autonomy --dry-run
```

프레임 수신 → 추론 → 궤적 → 속도 변환까지 전부 돌지만 **로봇에는 명령이 한 번도 나가지
않습니다.** 지시문을 넣고 Start를 누른 뒤 궤적 패널을 보세요.

확인할 것: 예측 경로가 앞을 향하고, 지시한 방향으로 대략 향하는가? 노이즈처럼 보이면
전처리나 체크포인트 문제이지 주행 문제가 아닙니다. 그 상태로는 뭘 조절해도 소용없습니다.

괜찮아 보이면 `--dry-run`을 떼고, **트인 곳에서, E-STOP에 손을 올려두고** 시작하세요.

---

## ⚠️ 주의사항

### `/control`은 마지막 명령을 계속 유지합니다

로봇은 새 명령이 올 때까지 **직전 명령을 계속 실행**합니다. 아래 안전장치가 전부 여기서
나옵니다.

- **E-STOP은 모델 프로세스를 거치지 않습니다.** 페이지가 SDK 서버의 `/control`로 0을
  직접 3번 보내고, 그 다음에야 모델에게 정지를 요청합니다. **모델이 죽어도 작동하는
  유일한 조작**이고, 죽었을 때가 바로 필요한 순간입니다. **Space**도 같습니다.
- 모델 프로세스는 **모든 종료 경로에서** 정지 명령을 보냅니다 — Stop, E-STOP, Ctrl+C,
  SIGTERM, 루프의 `finally`.
- 추론이나 통신 에러가 나면 **루프를 멈추고 로봇을 세웁니다.** 오래된 명령으로 계속
  달리지 않습니다.
- 일시정지 중에는 `/control`에 **아무것도 보내지 않습니다.**

### 전처리가 틀려도 에러가 안 납니다

96×96 리사이즈, 정규화, 컨텍스트 프레임 간격 — 이게 파인튜닝 때와 어긋나면 **예외가
발생하지 않고 주행 품질만 조용히 나빠집니다.** [omnivla_policy.py](policy/omnivla_policy.py)는
원본 `run_omnivla_edge.py`를 그대로 옮긴 것입니다. 여기를 고치는 건 모델을 바꾸는 것과
같다고 생각하세요.

특히 `--context-stride`: 기본값 1은 히스토리 프레임이 0.33초 간격이라는 뜻입니다.
파인튜닝 데이터의 프레임 간격이 이보다 촘촘했다면 맞춰야 합니다.

### 아직 실제 로봇에서 검증되지 않았습니다

mock 서버와 실제 브라우저로는 전 구간을 확인했지만, **진짜 로봇으로 주행해본 적은
없습니다.**

### 지연

카메라 프레임이 Agora를 통해 원격에서 오기 때문에 모델이 보는 화면은 이미 수백 ms
과거입니다. 원저자의 셋업(로봇에 직접 붙은 카메라)보다 불리한 조건입니다. 페이지가 추론
시간과 루프 시간은 보여주지만 **영상 자체의 지연은 측정하지 않습니다.**

arm-4′의 단계별 시간입니다. RTX 4070 SUPER · 34 tick · 예열 2 tick 제외 ·
`policy.check_ours`로 실측했습니다. **로봇이 빠진 수치**입니다 — 프레임 수신, 구동
명령 전송, 네트워크가 여기 없습니다. 측정 당시 같은 GPU에서 학습(`train_arm.py`)이
돌고 있었으므로 **경쟁 상태의 값**입니다. 비어 있는 GPU에서는 더 빠릅니다.

| 단계 | 중앙값 | p90 |
|---|---|---|
| 입력 준비 | 39 ms | 58 ms |
| 검출 (YOLOv8n) | 19 ms | 24 ms |
| CLIP heatmap | 14 ms | 28 ms |
| 선택 규칙 | 0.1 ms | 0.1 ms |
| 정책 forward | 29 ms | 30 ms |
| **tick 전체** | **102 ms** | **137 ms** |

333 ms 예산의 3분의 1이 안 되고, 34 tick 중 초과는 0건입니다. 첫 tick만 CUDA 커널
선택 때문에 483 ms가 걸립니다. 루프는 정지 상태로 시작하고 멈춰 있는 동안에는 정책을
호출하지 않으므로, 그 한 tick은 로봇이 서 있는 동안 지나갑니다.

### GPS

지금까지 기록된 어떤 세션에서도 GPS fix가 잡힌 적이 없습니다(fix가 없으면 로봇이
`latitude`/`longitude`를 1000으로 보냅니다). GPS 목표는 실험적입니다.

---

## 목표 지정 방식

| 방법 | modality id | 설명 |
|---|---|---|
| 웹 입력창, 또는 `--prompt` | 7 | **언어만. GPS 불필요 — 기본 선택지입니다.** |
| `--relative-goal 전방_m,좌측_m` | 4 | 현재 로봇 기준 좌표. pose 브랜치 테스트용 |
| `--goal-image 경로` | 6 | 목적지에서 찍은 1인칭 사진 |
| 지시문 + `--relative-goal` | 8 | pose와 언어 동시 사용 |
| `--goal-latlon 위도,경도` | 4 | GPS 목표. **실험적** |

위성 영상은 이 로봇에 없어서 두 슬롯 모두 검은 이미지가 들어가고, modality 0~3은 쓸 수
없습니다.

`--goal-latlon`이 실험적인 이유는 두 가지입니다. GPS fix가 아직 한 번도 안 잡혔고,
`/data`의 `orientation`이 나침반 방위각(도)이라고 **가정**했을 뿐 실측으로 확인하지
않았습니다. fix가 없으면 루프가 아예 시작을 거부합니다 — 수천 km 떨어진 목표를 향해
달리는 것보다 낫기 때문입니다.

### 지시문 작성법

텍스트는 CLIP ViT-B/32 텍스트 인코더를 통과합니다. 문장 전체를 임베딩 하나로 만들기
때문에 **단계별 지시를 해석하지 못합니다** ("3미터 직진 후 좌회전" 같은 건 안 됩니다).

**짧은 영어 명사구**가 모델이 학습된 형식입니다. 원본 예제도 `"blue trash bin"`입니다.
CLIP의 77토큰을 넘으면 조용히 잘립니다.

---

## 속도 캘리브레이션

모델은 **m/s, rad/s**로 속도를 뱉는데 `/control`은 **-1~1 정규화값**을 받습니다. 이걸
잇는 상수 두 개가 있고, 초기값은 placeholder `1.0`입니다.

수동 주행 때는 필요 없었습니다. 사람이 보고 조절하는 피드백 루프였으니까요. 모델에는
그런 루프가 없어서 스케일을 알려줘야 합니다.

**필수는 아닙니다.** placeholder로도 주행합니다. 틀렸을 때 치르는 비용:

- 둘 다 **같은 비율**로 틀림 → 속도만 빠르거나 느림. **경로 모양은 정상**
- 둘이 **다른 비율**로 틀림 → 직진 대비 회전량이 어긋나서 **커브를 계속 짧게 자르거나
  크게 돎**

두 번째 때문에 하는 겁니다. 로봇이 계속 오버스티어/언더스티어 하면 그때 측정하세요:

1. 주행을 멈추고 트인 곳으로 이동
2. **Drive straight 10s** → 이동 거리 측정
3. **Spin 10s** → 회전 수 세기
4. 두 값을 입력하고 **Save calibration**

타이머는 페이지가 아니라 **모델 프로세스가** 돌립니다. 브라우저 탭이 중간에 닫혀도
로봇이 계속 굴러가지 않게 하려는 겁니다. E-STOP으로 중단할 수 있고, 주행 중에는 버튼이
잠깁니다.

결과는 `policy/calibration.json`에 저장되어 다음 실행부터 자동 적용됩니다.
`--max-lin-mps` / `--max-ang-rps`로 덮어쓸 수 있습니다.

---

## 주행 데이터 기록

**Start Recording**을 누르면 됩니다. 기존 `/dataset/*` 엔드포인트를 그대로 써서
`dataset/sessions/<타임스탬프>/`에 `images/`, `control.jsonl`, `gps_imu.jsonl`을
남깁니다. 수동 주행 페이지가 만들던 것과 같은 형식입니다.

기록은 페이지가 아니라 **모델 루프가** 수행합니다. 그래야 저장된 프레임이 모델이 실제로
본 프레임이고, 짝지어진 명령이 실제로 나간 명령입니다. 페이지가 하면 자기 폴링 주기 때문에
프레임이 중복되거나 빠집니다.

`--record`를 붙이면 주행 시작과 동시에 기록합니다. 세션 업로드는
[upload_to_hf.py](upload_to_hf.py)를 쓰세요.

---

## 문제 해결

| 증상 | 원인 |
|---|---|
| 페이지에 `policy offline` | 터미널 2가 안 떠 있거나 죽음. 페이지는 SDK 서버가 서빙하므로 어차피 열립니다. E-STOP은 여전히 작동합니다. |
| 모델이 영상 스트림을 계속 기다림 | SDK 서버가 아직 프레임을 못 받음. 터미널 1의 Chrome/Agora 에러, `CHROME_EXECUTABLE_PATH` 확인. |
| `Bot unavailable for SDK` | 다른 사람이 로봇을 조종 중이거나, `MISSION_SLUG`가 설정됐는데 `/start-mission`을 안 함 |
| `--dry-run`에서 궤적이 노이즈 | 전처리가 파인튜닝 때와 다르거나 카메라 화각이 학습 분포 밖. `--context-stride`부터 확인 |
| 커브를 계속 짧게 자름 / 크게 돎 | [캘리브레이션](#속도-캘리브레이션) |
| 주행은 하는데 지시를 무시함 | 텔레메트리의 modality id가 7인지 확인, 더 짧고 구체적인 명사구로 시도 |
| `no kernel image is available` | GPU에 비해 torch가 오래됨. `pip install --upgrade --force-reinstall torch torchvision` — conda 패키지 말고 pip으로, 버전 고정 없이 |
| `hypercorn: command not found` / `ModuleNotFoundError` | `conda activate rover`를 안 했습니다. 프롬프트에 `(rover)`가 보이는지 확인하세요. hypercorn 잘못이 아닙니다 |
| 모델 쪽에 `400 Client Error` | 서버가 400을 돌려준 것. `curl -i http://127.0.0.1:8000/v2/front`로 `detail`을 보세요 — `Call /start-mission...`이면 **`.env`의 `MISSION_SLUG`를 지우고 서버 재시작**, `Failed to retrieve tokens`면 `SDK_API_TOKEN`/`BOT_SLUG`가 틀린 것 |
| 서버 로그의 `Running on http://0.0.0.0:8000` | 에러가 아니라 hypercorn이 정상 기동했다는 메시지입니다 |
| `No usable browser found` / 브라우저가 안 뜸 | `python -m playwright install chromium`을 빼먹었습니다 |
| 영상이 검게만 나옴 | Playwright 번들 Chromium에는 H.264 코덱이 없습니다. Google Chrome을 설치하거나 `.env`의 `CHROME_EXECUTABLE_PATH`로 지정하세요 |
| `conda: command not found` | `conda init bash` 후 터미널을 새로 안 열었거나 conda 미설치 → [0단계](#0-conda-설치-확인) |
| `CondaError: Run 'conda init' before 'conda activate'` | 같은 원인. 새 터미널을 열거나 `source ~/.bashrc` |
| `CondaValueError: prefix already exists` | 같은 이름의 환경이 이미 있습니다. `-n 다른이름`을 주세요 — 이름은 아무거나 되고 기존 환경은 그대로 남습니다 → [1단계](#1-conda-환경-만들기) |
| `requirements-all.txt`를 깔았는데 `ultralytics` / `open_clip` 없음 | 그 둘은 일부러 pip으로 안 깝니다. `models/omni_deps`를 복사하세요 → [omni_deps](#omni_deps는-pip로-설치하지-마세요) |
| 기존 환경에 깔았더니 torch가 깨짐 | 루트 `requirements.txt`의 `numpy==1.26.4`가 numpy를 다운그레이드한 것. 이 포크는 **새 환경**을 전제로 합니다 → [1단계](#1-conda-환경-만들기) |
| `conda env create`가 CLIP에서 실패 | 환경에 git이 없음(`environment.yml`이 깔아줍니다). 사내망이면 `git+https://` 접근 여부부터 확인 |
| `torch.cuda.is_available()`이 `False` | NVIDIA 드라이버(`nvidia-smi`) 확인 후 torch 재설치. **CPU로는 못 돕니다** |
| 레포 폴더에서만 `python`이 이상하게 동작 | pyenv를 쓰는 경우, 레포의 `.python-version`(`venv39`)이 conda보다 먼저 잡힙니다. 그 파일을 지우거나 pyenv를 끄세요 |
| 페이지가 404 / 캘리브레이션이 안 보임 | 레포 루트가 아닌 곳에서 실행. `pwd` 확인 후 `cd` |

---

## 실행 옵션 전체

보통은 아무것도 안 줘도 됩니다.

**정책** — arm-4′가 기본이라 아무것도 안 줘도 됩니다 → [정책 — arm-4′](#정책--arm-4)

| 옵션 | 기본값 | 설명 |
|---|---|---|
| (없음) | arm-4′ | 파인튜닝한 heatmap 4채널 정책 |
| `--tick-log` | `field_log/` | tick 로그와 heatmap 썸네일 |
| `--anchor-mode` | `crop_cos` | 기준 물체(B) 를 어떻게 찾는지. `heatmap_peak` 은 이전 동작입니다 — 현장에서 이상하면 되돌리는 용도 |

**기본**

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--ckpt` | `models/` 안의 정책별 기본값 | 체크포인트 경로 |
| `--server` | `http://localhost:8000` | SDK 서버 주소 |
| `--device` | `cuda:0` | CUDA 장치여야 함 |
| `--ui-port` | `8010` | `/state`, `/cmd` 포트 |
| `--no-ui` | 꺼짐 | 웹 없이 실행. 목표를 미리 줘야 함 |
| `--verbose` | 꺼짐 | 디버그 로그 |

**환경변수** — 기본 위치를 쓰면 하나도 필요 없습니다.

| 변수 | 기본값 | 용도 |
|---|---|---|
| `MODELS_DIR` | `<레포>/models` | 가중치 위치 |
| `OMNI_DEPS` | `<MODELS_DIR>/omni_deps` | ultralytics, open_clip |
| `OMNIVLA_REFS_ROOT` | `policy/refs` | 참조 코드·설정 위치 |
| `OMNIVLA_DATASET_ROOT` | `<MODELS_DIR>/frames` | 점검용 프레임 위치 |

**목표** — 전부 선택사항입니다. 지시문은 보통 웹에서 입력하고, `--prompt`는 입력창을
미리 채워둘 뿐입니다. `--autostart`나 `--no-ui`처럼 **아무도 타이핑할 수 없는 경우에만**
필수입니다.

**루프**

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--rate` | `3` | 제어 주기(Hz). 선택한 waypoint까지 도달할 시간도 같이 정해집니다 — 의도적으로 연동돼 있습니다 |
| `--context-stride` | `1` | 히스토리 프레임 간격(tick). 1이면 tick마다 한 장(약 0.33초 간격) |
| `--waypoint-index` | `4` | 8개 waypoint 중 몇 번째를 향해 갈지. 원본 값. 올리면 lookahead가 길어집니다 |

**안전**

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--dry-run` | 꺼짐 | 전부 돌리되 명령은 안 보냄 |
| `--autostart` | 꺼짐 | 즉시 주행 시작. 목표를 미리 줘야 함 |
| `--linear-cap` / `--angular-cap` | `0.5` | 정규화 명령 상한 |
| `--max-lin-mps` / `--max-ang-rps` | `calibration.json` | 측정된 속도 상수 덮어쓰기 |
| `--record` | 꺼짐 | 주행과 동시에 기록 시작 |

---

## 동작 원리

### 모델 입출력

`best.pth`는 `OmniVLA_edge` state dict입니다. ViNT 계열, 약 1억 400만 파라미터 —
EfficientNet-B0 인코더 3개, 4층 트랜스포머, 언어용 FiLM 브랜치. 원본 아키텍처에
`strict=True`로 로드됩니다.

입력 (전부 [omnivla_policy.py](policy/omnivla_policy.py)에서 만듭니다):

| 텐서 | shape | 내용 |
|---|---|---|
| `obs_img` | (1, 18, 96, 96) | 6프레임(현재+과거 5)을 채널로 concat |
| `goal_pose` | (1, 4) | `[y/0.1, -x/0.1, cos Δyaw, sin Δyaw]`, 로봇 좌표계, 단위 0.1m |
| `map_images` | (1, 9, 96, 96) | 위성_현재 + 위성_목표 + 현재 프레임 (위성 슬롯은 검은 이미지) |
| `goal_img` | (1, 3, 96, 96) | 1인칭 목표 이미지 |
| `goal_mask` | (1,) | modality id, 0~9 |
| `feat_text` | (1, 512) | CLIP ViT-B/32 텍스트 임베딩 |
| `current_img` | (1, 3, 224, 224) | FiLM 브랜치. 224는 2×2×1024=4096 헤드 때문에 고정 |

출력은 (8, 4) — waypoint 8개, 누적된 `(dx, dy)`(단위 0.1m)와 정규화된 `(cos, sin)`
heading. 로봇 좌표계 기준 **x=전방, y=좌측**입니다.

forward에서 `Tensor.get_device()`를 호출하는데 CPU에서는 -1을 반환하고 그대로 실패합니다.
**원본이 CUDA 전용이고 이 코드도 마찬가지입니다.**

### waypoint → 주행 명령

[control.py](policy/control.py)는 원본 변환식을 그대로 옮긴 것입니다. 8개 중 4번
waypoint를 골라 미터로 환산하고, `v = dx/dt`, `ω = arctan(dy/dx)/dt`를 계산한 뒤,
clip하고 회전 반경을 보존하는 리미터를 0.3 m/s, 0.3 rad/s에 겁니다.

랜덤 궤적 2000개로 원본 로직과 대조했고 최대 오차는 1e-16입니다.

**경로 추종(tracking) 코드는 없고, 필요하지도 않습니다.** waypoint가 로봇 기준 좌표라
드리프트할 전역 경로 자체가 없고, 궤적 전체를 매 tick마다 버리고 새 이미지로 다시
예측합니다. MPC와 같은 receding-horizon 구조로, 계획의 앞부분만 실행하는 셈입니다.
피드백 루프가 모델 자신입니다.

대신 포기하는 것: **8개 중 7개 waypoint**, 그리고 거기 담긴 곡률 정보입니다. 급커브를
안쪽으로 자를 수 있습니다. 실주행에서 그게 보이면 진짜 트래커를 붙이기 전에
`--waypoint-index`부터 올려보세요 — 모델이 이 컨트롤러와 함께 튜닝됐기 때문에 바꾸면
학습 분포에서 벗어납니다.

---

## 파일 구성

| 파일 | 역할 |
|---|---|
| [policy/model_omnivla_edge.py](policy/model_omnivla_edge.py) | [NHirose/OmniVLA](https://github.com/NHirose/OmniVLA)에서 그대로 가져옴(MIT). **수정 금지** |
| [policy/omnivla_policy.py](policy/omnivla_policy.py) | 체크포인트 로드, 전처리, 모달리티 선택, forward |
| [policy/control.py](policy/control.py) | waypoint → (v, ω) → 정규화 명령, 캘리브레이션 저장 |
| [policy/rover_client.py](policy/rover_client.py) | SDK 서버용 HTTP 클라이언트 |
| [policy/run_autonomy.py](policy/run_autonomy.py) | 제어 루프, 안전장치, `/state`·`/cmd` 서버 |
| [policy/check_model.py](policy/check_model.py) | 오프라인 체크포인트 점검 (`--upstream` 경로) |
| [policy/ours_policy.py](policy/ours_policy.py) | arm-4′·arm-1 정책. 파싱 → 검출 → CLIP → heatmap 채널 |
| [policy/check_ours.py](policy/check_ours.py) | arm-4′ 로봇 연결 전 점검 3종 |
| [policy/visualize_pipeline.py](policy/visualize_pipeline.py) | 사진 1장의 파이프라인 전 단계를 한 장으로 그림 |
| [policy/refs/](policy/refs/) | 참조 저장소에서 그대로 가져온 코드·설정. **수정 금지** |
| `models/` | 가중치. 빈 폴더로 커밋되고 내용은 gitignore → [모델 넣기](#2-모델-넣기) |
| `field_log/` | tick 로그와 heatmap 썸네일. gitignore |
| [static/autonomy_control.html](static/autonomy_control.html) | 조작 페이지 (SDK 서버가 서빙) |
| [environment.yml](environment.yml) | conda 환경 하나 (Python 3.11 + [requirements.txt](requirements.txt) + [policy/requirements.txt](policy/requirements.txt)). 이름 기본값은 `rover`, `-n`으로 덮어씁니다 |
| [requirements-all.txt](requirements-all.txt) | 위 둘 + `upload_to_hf.py`를 한 번에. conda 파일 없이 새 환경에 깔 때 → [1단계](#1-conda-환경-만들기) |
| `policy/calibration.json` | 측정된 속도 상수. 페이지가 생성, gitignore됨 |

`main.py`를 비롯한 SDK 서버 코드는 수정하지 않았습니다. 예외는
[browser_service.py](browser_service.py) 하나입니다.

### 브라우저 드라이버 (pyppeteer → Playwright)

원래 이 포크는 헤드리스 브라우저를 **pyppeteer**로 몰았습니다. pyppeteer는 유지보수가
끊긴 패키지입니다 — `urllib3<2`와 `websockets<11`을 강제로 고정하고, Python 3.13용 휠이
없습니다. 그래서 요즘 환경에서 설치부터 깨집니다.

[공식 레포](https://github.com/frodobots-org/earth-rovers-sdk)는 이미 **Playwright**로
넘어갔고, 이 포크도 거기에 맞췄습니다. [requirements.txt](requirements.txt)는 이제 공식
레포와 **완전히 동일**합니다.

바뀐 것:

- `pyppeteer==2.0.0` → `playwright==1.60.0` + `wsproto==1.2.0`
- **Python 버전 제약이 사라졌습니다.** 서버를 3.9에 묶어두던 게 pyppeteer였습니다. 그래서
  conda 환경도 서버/모델 **두 개에서 하나(`rover`)로 합쳤습니다.**
- `CHROME_EXECUTABLE_PATH`가 **선택사항**이 됐습니다. `CHROME_EXECUTABLE_PATH` → 설치된
  Google Chrome → Playwright 번들 Chromium 순으로 찾습니다. macOS 경로가 기본값으로
  박혀있던 문제도 같이 사라졌습니다.
- 브라우저가 죽으면 **자동으로 다시 띄웁니다**(`_run`의 재연결). 단 `/control`과 `/speak`은
  재시도하지 않습니다 — 정지 명령이 뒤늦게 두 번 나가면 안 되니까요.
- 첫 프레임을 기다릴 때 `<video>` 엘리먼트가 아니라 **RTM 준비 상태**를 기다립니다. 카메라가
  꺼져 있어도 조작과 텔레메트리는 살아있어야 하기 때문입니다.

`main.py`가 쓰는 메서드(`data`, `front`, `rear`, `send_message`, `speak`,
`take_screenshot`)와 페이지 쪽 JS 계약(`window.sendMessage`, `getLastBase64Frame`,
`window.rtm_data`)은 그대로라, `main.py`와 `static/`은 건드리지 않았습니다.

### 모델 프로세스 HTTP 인터페이스

페이지가 쓰는 건 이 두 개가 전부입니다. 외부 에이전트가 로봇을 몰 때도 이거면 됩니다.

```bash
curl http://localhost:8010/state          # 페이지에 표시되는 모든 값

curl -X POST http://localhost:8010/cmd -H 'Content-Type: application/json' \
     -d '{"action":"prompt","prompt":"the red door"}'
```

`action`은 `start`, `stop`, `estop`, `prompt`, `record`(`{"on": true}`),
`calibrate_move`(`{"kind","value","seconds"}`), `calibrate_set`
(`{"max_lin_mps","max_ang_rps"}`) 중 하나입니다.

---

## SDK 엔드포인트

모델 프로세스가 쓰는 것들입니다. 직접 호출할 일은 거의 없습니다.

| 엔드포인트 | 용도 |
|---|---|
| `POST /control` | `{"command":{"linear":0.5,"angular":0,"lamp":0}}` — 각 -1~1 |
| `GET /data` | 배터리, GPS, IMU 등 로봇이 RTM으로 방송하는 원본 값 |
| `GET /v2/front` | 전방 카메라 프레임 (base64) + 타임스탬프 |
| `GET /v2/rear` | 후방 카메라 (zero 기종만) |
| `POST /dataset/start` · `log-frame` · `stop` · `GET /status` | 주행 데이터 기록 |
| `POST /speak` | `{"text":"..."}` — 로봇 스피커로 TTS 재생 |
| `GET /missions` · `POST /start-mission` · `/end-mission` | 미션 API |
| `POST /interventions/start` · `/end` · `GET /history` | 개입 기록 API |

전체 명세는 [원본 레포](https://github.com/frodobots-org/earth-rovers-sdk)를 참고하세요.

---

## 출처

모델과 추론 로직은 [NHirose/OmniVLA](https://github.com/NHirose/OmniVLA)(MIT)에서
가져왔습니다. Noriaki Hirose, Catherine Glossop, Dhruv Shah, Sergey Levine 저,
[OpenVLA-OFT](https://openvla-oft.github.io/) 기반. 논문:
[arXiv:2509.19480](https://arxiv.org/abs/2509.19480).

SDK 본체는 [frodobots-org/earth-rovers-sdk](https://github.com/frodobots-org/earth-rovers-sdk)
— Michael Cho, Santiago Pravisani, Esteban Fuhrmann.
