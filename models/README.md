# `models/`

이 폴더는 **빈 상태로 커밋**돼 있습니다. git clone 하면 이 파일 하나만 있습니다.
가중치는 너무 커서 깃에 올릴 수 없으니, 다른 컴퓨터의 `models/` 내용을 **여기에 그대로
복사**하세요. 골라 담을 필요 없고, 폴더를 새로 만들 필요도 없습니다.

전체 절차는 레포 루트의 [README.md](../README.md#2-모델-넣기)에 있습니다.

## 옮기기

```bash
# 보내는 쪽
cd ~/forodobot_semantic_nav/models
cp -r . /media/$USER/<USB이름>/models/
sync                                    # 뽑기 전에. 안 하면 파일이 잘릴 수 있습니다

# 받는 쪽
cd ~/forodobot_semantic_nav
cp -r /media/$USER/<USB이름>/models/. models/
(cd models && sha256sum -c SHA256SUMS)  # OK 다섯 줄
```

`scp`도 같습니다.

```bash
scp -r ~/forodobot_semantic_nav/models/. <노트북>:~/forodobot_semantic_nav/models/
```

## 들어있는 것

```
models/
  arm4p_s0_latest.pth            415M   정책 (arm-4′)          ← 필수
  act4_cl6159_s2.pt               55M   CLIP 어댑터             ← 필수
  full_H1_linear_lr0.001_s0.pt   1.1M   국소화 헤드             ← 필수
  yolov8n.pt                     6.3M   검출기                  ← 필수
  omni_deps/                      12M   ultralytics, open_clip  ← 필수
  arm1_latest.pth                418M   대조군 (--arm1)         ← 현장 A/B에 필요
  frames/                        0.5M   점검용 프레임           ← 점검에 필요
  clip/                          338M   CLIP ViT-B/32 캐시      ← 인터넷 있으면 생략 가능
  hf/                            571M   CLIP ViT-B/16 캐시      ← 인터넷 있으면 생략 가능
  SHA256SUMS                            복사 검증용 목록
```

전부 1.8 GB입니다. 인터넷이 되는 컴퓨터라면 `clip/`과 `hf/`를 빼서 906 MB로 줄일 수
있습니다. 그 둘은 첫 실행 때 자동으로 받습니다. 용량이 문제가 아니면 **전부 복사하는
쪽이 낫습니다** — 첫 실행이 빠르고, 현장에서 네트워크가 끊겨도 돕니다.

`omni_deps/`는 pip로 설치하지 말고 **디렉토리째** 복사하세요. `--no-deps`로 설치된
것이라, `pip install ultralytics`를 하면 자기가 쓰던 torch를 끌고 와서 지금 설치된
torch를 덮어버립니다.

## 제대로 들어갔는지 확인

```bash
python -m policy.check_ours
```

파일이 빠져 있으면 무엇이 없는지 목록으로 알려줍니다. `port PASS`가 나와야 로봇에
붙일 수 있습니다.
