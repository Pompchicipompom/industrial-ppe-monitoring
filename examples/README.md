# Примеры конфигов

Файлы в этом каталоге предназначены только для проверки запуска и
быстрой демонстрации. Для оценки качества используйте конфиги из
`configs/`.

## Hardhat

```bash
python main.py --config examples/example_config_hardhat.yaml \
    --source input/demo.mp4 --output runs/example_hardhat
```

## Vest

```bash
python main.py --config examples/example_config_vest.yaml \
    --source input/demo.mp4 --output runs/example_vest
```

## Что нужно для запуска

1. Поместить веса моделей в `models/`
   (см. [../docs/model_files.md](../docs/model_files.md)).
2. Положить демонстрационное видео в `input/demo.mp4` (или указать
   собственный путь через `--source`).
3. Установить зависимости: `pip install -r ../requirements.txt`.
