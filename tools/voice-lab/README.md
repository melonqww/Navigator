# Голосовая лаборатория

Локальная проверка WAV и генерации речи через Qwen3-TTS 0.6B Base. API-ключ не требуется. Пользовательское аудио обрабатывается локально; сетевой доступ нужен для первоначальной установки и загрузки модели.

## Быстрые проверки

Из корня репозитория, Node.js 24.13+ и Python 3.11:

```powershell
npm test
py -3.11 -m unittest discover -s tools/voice-lab/test -p 'test_*.py' -v
npm run voice:lab -- phrases
```

Эти команды не требуют ML-пакетов, весов или GPU. Node-инструмент не имеет внешних зависимостей.

## Установка для генерации на GPU

Нужны Windows, Python 3.11, драйвер NVIDIA, доступ в интернет и несколько гигабайт свободного места. Команды выполняются из корня репозитория. Среда и веса сохраняются в исключённой из Git папке `data/`.

```powershell
py -3.11 -m venv data/voice-runtime
data/voice-runtime/Scripts/python.exe -m pip install torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
data/voice-runtime/Scripts/python.exe -m pip install -r tools/voice-lab/requirements.lock
data/voice-runtime/Scripts/python.exe tools/voice-lab/local.py doctor
data/voice-runtime/Scripts/python.exe tools/voice-lab/local.py download
data/voice-runtime/Scripts/python.exe tools/voice-lab/local.py load-check
```

`requirements.txt` содержит основные зависимости, `requirements.lock` фиксирует их полный набор для чистой установки. GPU-измерения в основном README получены на другой совместимой конфигурации: Python 3.13 / PyTorch 2.7.1 + CUDA 11.8. Чистая конфигурация выше ещё не прошла полный GPU-прогон; её совместимость необходимо проверить командами `load-check` и `runtime-smoke`.

`download` получает конфигурации и два файла весов из официального репозитория Qwen на фиксированной ревизии. Веса загружаются частями с возобновлением и проверкой SHA-256. При прерывании повтори ту же команду. Отдельный `download_weights.py` получает только веса и не заменяет первоначальную загрузку конфигураций.

Остальные команды работают offline. Используется PyTorch SDPA. Для проверенного пути не требуется компиляция FlashAttention или вызов SoX.

## Подготовка записи

Прочитай [короткий текст](../../fixtures/phrases/reference-short.ru.txt) обычным голосом и сохрани `data/samples/my-voice.wav`. [Подробная инструкция](../../fixtures/phrases/recording.ru.md).

Первый инструмент принимает WAV PCM 16 бит, 16–96 кГц, моно или стерео. Для синтеза — 3–30 секунд. Если произнесённые слова отличаются от образца, сохрани точную расшифровку отдельным TXT. Другие форматы нужно конвертировать, а не переименовывать.

```powershell
npm run voice:lab -- inspect --sample data/samples/my-voice.wav
```

Проверяются формат, длительность, тишина, низкий уровень и перегрузка. Разборчивость, эхо и узнаваемость требуют прослушивания. Каталог окружения сам по себе не подтверждает наличие установленных ML-зависимостей — для этого используй Python-команду `doctor`.

## Новая фраза своим голосом

```powershell
data/voice-runtime/Scripts/python.exe tools/voice-lab/local.py synthesize --sample data/samples/my-voice.wav --transcript fixtures/phrases/reference-short.ru.txt --phrase turn-right --consent
```

`--consent` подтверждает право использовать запись. Микрофон команда не включает, аудио в интернет не отправляет. Другие идентификаторы фраз перечислены в `npm run voice:lab -- phrases`.

Результат каждого запуска — отдельная папка `data/voice-lab/local-…` с `result.wav` и `report.json`. Отчёт содержит ревизию модели, времена загрузки и синтеза, длительность результата и выделенную GPU-память. Сходство голосов автоматически не оценивается. Предел в 512 выходных токенов ограничивает генерацию; полноту фразы нужно проверить прослушиванием.

При прерывании процесса статус `running` означает незавершённый эксперимент. Автоматических повторов и устойчивой фоновой очереди в этой лаборатории нет.

## Проверки GPU-среды

Пять тестов с настоящими аудиобиблиотеками и подставным генератором, без загрузки модели:

```powershell
data/voice-runtime/Scripts/python.exe -X utf8 -m unittest discover -s tools/voice-lab/runtime-test -v
```

Запуск настоящей модели:

```powershell
data/voice-runtime/Scripts/python.exe tools/voice-lab/local.py runtime-smoke
```

`runtime-smoke` использует синтетический тон и искусственную расшифровку, ограничивая выход 32 токенами. Проверяется путь вычислений и корректность WAV; это не оценка качества клонирования. Выход сохраняется отдельно как `technical-output.wav`.

## Источники

- [Официальный Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- [Веса 0.6B Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)

Сторонние веса и библиотеки сохраняют свои лицензии и авторство. Автор проекта — **melonqww**.
