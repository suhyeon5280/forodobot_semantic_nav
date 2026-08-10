# Earth Rovers SDK — OmniVLA-edge 자율주행

[frodobots-org/earth-rovers-sdk](https://github.com/frodobots-org/earth-rovers-sdk)
포크. 파인튜닝한 [OmniVLA-edge](https://github.com/NHirose/OmniVLA) 체크포인트로
Earth Rover가 스스로 주행합니다. **웹페이지에 목적지를 글로 적으면 모델이 운전합니다.**

터미널 두 개는 처음에 한 번 띄우고 그 뒤로는 건드리지 않습니다. 조작은 전부 웹에서 합니다.

---

## 목차

- [전체 흐름 한눈에](#전체-흐름-한눈에)
- [구조](#구조)
- [준비물](#준비물)
- [최초 1회 설정](#최초-1회-설정)
- [업데이트 받기](#업데이트-받기)
- [실행](#실행)
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

---

## 전체 흐름 한눈에

**서버와 모델은 같은 컴퓨터 한 대**에서 터미널 두 개로 돕니다. 각 단계 설명은 아래 링크를
따라가세요.

**conda 환경은 하나(`rover`)뿐입니다.** 두 터미널 다 같은 환경을 씁니다.

```bash
# ── 최초 1회 ────────────────────────────────────────────────────────────
cd ~/frodobot_server-omnivla-edge-autonomy       # 자기 경로로
conda env create -f environment.yml              # rover (Python 3.11)
conda activate rover
python -m playwright install chromium            # 빼먹으면 브라우저가 안 뜹니다
cp .env.sample .env && vi .env                   # MISSION_SLUG 줄 삭제!
curl -L -o best.pth https://github.com/.../best.pth
python -m policy.check_model --ckpt best.pth     # 오프라인 점검

# ── 매번 ────────────────────────────────────────────────────────────────
# 터미널 1 — SDK 서버 (다른 기기에서 페이지를 열려면 --bind 0.0.0.0:8000)
cd ~/frodobot_server-omnivla-edge-autonomy && conda activate rover
hypercorn main:app

# 터미널 2 — 모델 (새 터미널, 같은 환경)
cd ~/frodobot_server-omnivla-edge-autonomy && conda activate rover
python -m policy.run_autonomy --ckpt best.pth --dry-run   # 첫 주행은 반드시 --dry-run

# 브라우저: http://localhost:8000/static/autonomy_control.html
```

자세히: [최초 1회 설정](#최초-1회-설정) · [실행](#실행) · [첫 주행](#첫-주행) ·
[문제 해결](#문제-해결)

---

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
        /state,/cmd  │    best.pth ─ OmniVLA-edge 추론       │
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
- **`best.pth`** — 레포에 없습니다. [릴리스에서 받으세요](#2-체크포인트-받기).

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

먼저 레포 루트로 가서, 여기가 맞는지부터 확인하세요. 아래 모든 명령의 기준점입니다.

```bash
cd ~/frodobot_server-omnivla-edge-autonomy    # 자기 경로로

pwd                    # 지금 위치
ls main.py policy/ environment.yml
```

세 개 다 찍히면 제대로 온 겁니다. `No such file or directory`가 뜨면 아직 레포 밖입니다.

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

**환경은 하나면 됩니다.** 서버와 모델이 같이 들어갑니다.

```bash
conda env create -f environment.yml     # rover (Python 3.11)
conda activate rover
```

프롬프트가 `(rover)`로 바뀝니다. torch, CLIP, EfficientNet을 받느라 몇 GB, 몇 분 걸립니다.

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

환경을 고쳐 만들려면:

```bash
conda env update -f environment.yml --prune   # yml 변경분만 반영
conda env remove -n rover                     # 통째로 지우고 다시 create
```

### 2. 체크포인트 받기

`best.pth`는 약 415MB로 GitHub의 파일당 100MB 제한을 넘습니다. 그래서 커밋하지 않고
**릴리스 첨부 파일**로 올려두었습니다. 릴리스 첨부 파일은 git 히스토리 밖에 있어서
`git clone`은 가볍게 유지되고, 가중치만 따로 받습니다.

```bash
curl -L -o best.pth \
  https://github.com/minsong0206/frodobot_server/releases/download/omnivla-v1/best.pth
```

레포 루트에 두세요. 다른 태그로 올렸다면 URL을 맞춰주세요.

```bash
ls -lh best.pth        # 415M 근처면 정상. 몇 KB면 다운로드 실패(HTML 에러 페이지)
```

### 3. `.env` 만들기

이 파일이 없으면 **서버가 인증 단계에서 죽습니다.** 레포 루트에 `.env`라는 이름으로
만드세요.

```bash
cp .env.sample .env
```

복사했다면 **`.env.sample`에 들어있는 `MISSION_SLUG` 줄을 반드시 지우세요.** 넣어두면 미션
모드가 되어 모든 엔드포인트가 `/start-mission`을 먼저 요구합니다. 자유 주행하려면 아예
빼야 합니다.

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

서버도 로봇도 필요 없습니다. `(rover)` 환경에서 레포 루트에 있으면 바로 실행됩니다.

```bash
python -m policy.check_model --ckpt best.pth
```

`state_dict matched strict=True`가 뜨고, 모달리티별 waypoint와 추론 시간이 출력되면
정상입니다 (RTX 4070 SUPER 기준 약 22ms/frame — 3Hz에 여유가 많습니다).

state_dict가 안 맞으면 그 체크포인트는 OmniVLA-edge 모델이 아니고, 아래 내용이 전부
무의미합니다.

실제 프레임으로 돌려보려면:

```bash
python -m policy.check_model --ckpt best.pth \
    --image dataset/sessions/<세션>/images/<프레임>.jpg \
    --prompt "the blue trash bin"
```

여기까지 끝나면 최초 설정은 끝입니다. 이후로는 [실행](#실행)의 터미널 두 개만 반복합니다.

---

## 업데이트 받기

로봇을 돌리는 컴퓨터가 개발하는 컴퓨터와 다르다면, 이미 `git clone` 해둔 쪽에서는 이렇게
갱신합니다. **`git pull`만으로는 부족합니다** — 의존성이 바뀌었으면 환경도 같이 갱신해야
합니다.

```bash
cd ~/frodobot_server-omnivla-edge-autonomy
git pull

conda activate rover
conda env update -f environment.yml --prune     # requirements가 바뀌었을 때
python -m playwright install chromium           # 브라우저가 아직 없다면
```

`git pull`이 로컬 변경 때문에 막히면, 그 컴퓨터에서 고친 게 없는 경우엔 이걸로 덮어씁니다:

```bash
git fetch origin && git reset --hard origin/main
```

`.env`, `best.pth`, `dataset/`은 gitignore라 pull이 건드리지 않습니다. 그대로 남습니다.

> **pyppeteer 시절에 받아둔 클론이라면** — 서버가 Playwright로 바뀌었으니 환경을 새로 만드는
> 게 깔끔합니다. 예전 `rover-sdk`/`rover-policy` 환경은 지워도 됩니다.
>
> ```bash
> conda env remove -n rover-sdk
> conda env remove -n rover-policy
> conda env create -f environment.yml
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
cd ~/frodobot_server-omnivla-edge-autonomy
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
cd ~/frodobot_server-omnivla-edge-autonomy
conda activate rover
python -m policy.run_autonomy --ckpt best.pth
```

프롬프트가 `(rover)`인지 확인하세요. `(base)`인 채로 실행하면
`ModuleNotFoundError: No module named 'torch'`가 납니다 — activate를 빼먹은 겁니다.

서버의 헤드리스 브라우저가 로봇의 Agora 채널에 붙어 첫 프레임을 뱉을 때까지 최대 1분
기다립니다. `operator UI: ...` 로그가 뜨면 준비 완료입니다.

**첫 주행이라면 여기서 `--dry-run`을 붙이세요** → [첫 주행](#첫-주행)

### activate 없이 한 줄로 (선택)

`conda run`을 쓰되 **`--no-capture-output`을 꼭 붙이세요.** 없으면 출력이 버퍼링돼서
`operator UI: ...` 같은 로그가 실시간으로 안 보입니다.

```bash
cd ~/frodobot_server-omnivla-edge-autonomy
conda run --no-capture-output -n rover hypercorn main:app                             # 터미널 1
conda run --no-capture-output -n rover python -m policy.run_autonomy --ckpt best.pth # 터미널 2
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
cd ~/frodobot_server-omnivla-edge-autonomy && conda activate rover
hypercorn main:app

# 터미널 2
cd ~/frodobot_server-omnivla-edge-autonomy && conda activate rover
python -m policy.run_autonomy --ckpt best.pth

# 브라우저: http://localhost:8000/static/autonomy_control.html
```

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
cd ~/frodobot_server-omnivla-edge-autonomy
conda activate rover
python -m policy.run_autonomy --ckpt best.pth --dry-run
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
| `conda env create`가 CLIP에서 실패 | 환경에 git이 없음(`environment.yml`이 깔아줍니다). 사내망이면 `git+https://` 접근 여부부터 확인 |
| `torch.cuda.is_available()`이 `False` | NVIDIA 드라이버(`nvidia-smi`) 확인 후 torch 재설치. **CPU로는 못 돕니다** |
| 레포 폴더에서만 `python`이 이상하게 동작 | pyenv를 쓰는 경우, 레포의 `.python-version`(`venv39`)이 conda보다 먼저 잡힙니다. 그 파일을 지우거나 pyenv를 끄세요 |
| 페이지가 404 / 캘리브레이션이 안 보임 | 레포 루트가 아닌 곳에서 실행. `pwd` 확인 후 `cd` |

---

## 실행 옵션 전체

보통은 `--ckpt`만 있으면 됩니다.

**기본**

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--ckpt` | `best.pth` | 체크포인트 경로 |
| `--server` | `http://localhost:8000` | SDK 서버 주소 |
| `--device` | `cuda:0` | CUDA 장치여야 함 |
| `--ui-port` | `8010` | `/state`, `/cmd` 포트 |
| `--no-ui` | 꺼짐 | 웹 없이 실행. 목표를 미리 줘야 함 |
| `--verbose` | 꺼짐 | 디버그 로그 |

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
| [policy/check_model.py](policy/check_model.py) | 오프라인 체크포인트 점검 |
| [static/autonomy_control.html](static/autonomy_control.html) | 조작 페이지 (SDK 서버가 서빙) |
| [environment.yml](environment.yml) | `rover` conda 환경 하나 (Python 3.11 + [requirements.txt](requirements.txt) + [policy/requirements.txt](policy/requirements.txt)) |
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
