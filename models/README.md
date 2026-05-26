# Файлы весов моделей

Веса детекторов не включены в репозиторий. После клонирования поместите
их в этот каталог.

## Откуда взять веса

Веса распространяются отдельно через Google Drive:

**[Открыть папку на Google Drive](https://drive.google.com/drive/folders/1YmBQYMUwpaXqmMdpY5acyaaalxJCVmNv)**

В папке размещён архив `google_drive_bundle.rar`. После распаковки
переместите `.pt`-файлы в этот каталог. Подробности по распаковке —
в [главном README](../README.md#веса-моделей-и-демонстрационные-видео-google-drive).

## Ожидаемые файлы

| Имя файла | Назначение |
| --- | --- |
| `hardhat_detection_yolo11_200_epochs_best_02032025.pt` | основной hardhat-детектор |
| `helmet_vest_repo_best.pt` | vest-детектор с прямым классом `NO-Safety Vest` |
| `hardhat_binary_best.pt` *(опционально)* | бинарный hardhat-детектор |

Если используются другие имена, поправьте `model.weights_path` в YAML.

## Подробнее

См. [../docs/model_files.md](../docs/model_files.md).
