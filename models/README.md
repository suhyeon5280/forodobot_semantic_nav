# `models/`

이 폴더는 **빈 상태로 커밋**돼 있습니다. git clone 하면 이 파일 하나만 있습니다.
가중치는 너무 커서 깃에 올릴 수 없으니, 아래 파일들을 **이 폴더에 그대로 복사**하세요.
새 폴더를 만들 필요는 없습니다.

전체 절차는 레포 루트의 [README.md](../README.md#2-모델-넣기)에 있습니다.

## 넣어야 하는 것

```
models/
  arm4p_s0_latest.pth            415M   정책 (arm-4′)          ← 필수
  act4_cl6159_s2.pt               55M   CLIP 어댑터             ← 필수
  full_H1_linear_lr0.001_s0.pt   1.1M   국소화 헤드             ← 필수
  yolov8n.pt                     6.3M   검출기                  ← 필수
  omni_deps/                      12M   ultralytics, open_clip  ← 필수
  arm1_latest.pth                418M   대조군 (--arm1)         ← 선택
  frames/                        1.5M   검증용 프레임           ← 선택
  clip/                          354M   CLIP ViT-B/32 캐시      ← 오프라인용
  hf/                            571M   CLIP ViT-B/16 캐시      ← 오프라인용
```

`omni_deps/`는 pip로 설치하지 말고 **디렉토리째 복사**하세요. `--no-deps`로 설치된
것이라, 그냥 `pip install ultralytics`를 하면 자기가 쓰던 torch를 끌고 와서 지금
설치된 torch를 덮어버립니다.

`clip/`과 `hf/`는 인터넷이 있으면 필요 없습니다. 첫 실행에 자동으로 받습니다.
현장에서 네트워크가 없을 거면 미리 복사해 두세요.

## 제대로 들어갔는지 확인

```bash
python -m policy.check_ours --skip-regression --skip-timing
```

파일이 빠져 있으면 무엇이 없는지 목록으로 알려줍니다.
