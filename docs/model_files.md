# Файлы весов моделей

Веса детекторов не включены в репозиторий из-за их размера и условий
распространения. Их необходимо получить отдельно и поместить в каталог
`models/`.

## Какие веса нужны

| Задача | Назначение | Ожидаемое имя файла |
| --- | --- | --- |
| no_hardhat | основной детектор `person / head / hardhat` | `hardhat_detection_yolo11_200_epochs_best_02032025.pt` |
| no_hardhat (опционально) | бинарная модель `with_hard_hat / without_hard_hat` | `hardhat_binary_best.pt` |
| no_vest | детектор с прямым классом `NO-Safety Vest` | `helmet_vest_repo_best.pt` |
| person fallback (опционально) | универсальный COCO-детектор | `yolov8s.pt` (Ultralytics автоматически скачает при необходимости) |

## Куда положить веса

```
public_repo_clean/
├── models/
│   ├── hardhat_detection_yolo11_200_epochs_best_02032025.pt
│   ├── helmet_vest_repo_best.pt
│   └── ...
```

Имена файлов совпадают с теми, что указаны в production-конфигах. Если
вы используете другие имена, поправьте `model.weights_path` в YAML.

## Как прописать путь в конфиге

Ключи, которые отвечают за пути к весам:

```yaml
model:
  weights_path: "models/hardhat_detection_yolo11_200_epochs_best_02032025.pt"
  person_fallback_weights_path: "yolov8s.pt"
  binary_weights_path: "models/hardhat_binary_best.pt"
```

В `production_vest.yaml` `weights_path` указывает на vest-детектор.

## Как проверить, что веса доступны

```bash
ls -la models/
```

или (Windows PowerShell):

```powershell
Get-ChildItem models
```

Должны присутствовать файлы `.pt`, размер которых соответствует
исходным весам (как правило, единицы — десятки мегабайт).

## Почему веса не включены

- бинарные веса дают значительный объём, не приспособленный для
  GitHub-репозитория;
- условия распространения весов зависят от их источника (включая
  публичные SH17-веса и собственно дообученные модели);
- репозиторий должен оставаться чисто текстовым, исключающим случайные
  бинарные конфликты при cross-platform работе.

В рамках ВКР архив с весами передаётся отдельно (через файловое
хранилище, релиз-ассет или Git LFS) и не публикуется в `git push`.
