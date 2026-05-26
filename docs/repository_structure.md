# Структура репозитория

```
public_repo_clean/
├── README.md                       краткое описание, запуск, результаты
├── requirements.txt                runtime-зависимости
├── .gitignore                      исключения для git
├── .gitattributes                  нормализация переводов строк
├── main.py                         CLI-точка входа
│
├── ppe_monitoring/                 основной пакет
│   ├── __init__.py
│   ├── config.py                   загрузка YAML-конфига и значения по умолчанию
│   ├── detector.py                 YOLO-обёртка, нормализация классов, ROI-инференс
│   ├── event_consolidator.py       EventConsolidatorV3 (анкер-история, multi-track)
│   ├── event_logic.py              покадровая временная логика nohardhat/no_vest
│   ├── geometry.py                 ROI, IoU, утилиты bbox
│   ├── metrics_constants.py        константы колонок frame_metrics.csv
│   ├── motion.py                   motion detector, frame sampler, inference gate
│   ├── person_confirmation.py      мягкое подтверждение person
│   ├── person_head_confirmation.py подтверждение person по head/hardhat
│   ├── pipeline.py                 связующий runner pipeline
│   ├── profiler.py                 замер таймингов стадий loop'а
│   ├── rtsp_health.py              watchdog RTSP-источника
│   ├── tracker.py                  PersonTracker по IoU + перенос треков
│   ├── types.py                    dataclass-структуры детекций / событий
│   ├── video_id.py                 утилита определения video_id
│   └── visualization.py            рендер bbox / плашек / кириллица через PIL
│
├── configs/                        production-конфиги
│   ├── production_hardhat.yaml
│   ├── production_vest.yaml
│   └── example_video.yaml
│
├── tools/                          сервисные скрипты
│   ├── eval_events.py              event-level evaluator (P/R/F1)
│   └── consolidate_events.py       пост-обработка events.csv через V3
│
├── examples/                       примеры конфигурации
│   ├── example_config_hardhat.yaml
│   ├── example_config_vest.yaml
│   └── README.md
│
├── docs/                           документация
│   ├── architecture.md             архитектура pipeline
│   ├── event_logic.md              временная логика + EventConsolidatorV3
│   ├── evaluation.md               методика оценки
│   ├── results_summary.md          итоговые результаты
│   ├── model_files.md              как получить и подключить веса
│   └── repository_structure.md     этот файл
│
├── models/                         каталог для весов (содержимое не коммитится)
│   ├── README.md                   инструкция по весам
│   └── .gitkeep
│
├── input/                          входные видео (не коммитятся)
│   └── .gitkeep
├── output/                         выходные артефакты (не коммитятся)
│   └── .gitkeep
└── runs/                           запуски (не коммитятся)
    └── .gitkeep
```

## Назначение каталогов

| Каталог | Назначение |
| --- | --- |
| `ppe_monitoring/` | программный модуль системы (Python-пакет) |
| `configs/` | финальные production-конфиги и пример |
| `tools/` | вспомогательные CLI-утилиты |
| `examples/` | минимальные примеры конфига и инструкции по запуску |
| `docs/` | техническая документация |
| `models/` | каталог для весов; веса не коммитятся в git |
| `input/` | каталог для входных видео; видео не коммитятся |
| `output/` | каталог для выходных артефактов одиночного запуска |
| `runs/` | каталог для именованных запусков |
