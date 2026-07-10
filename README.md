# Orchestra Runtime

Локальный runtime для запуска GGUF-моделей. Go-сервер с OpenAI-совместимым API,
использует llama.cpp через CGo. Часть экосистемы [Operium Orchestra](https://operium.ru).

*Local runtime for GGUF models: Go server + llama.cpp via CGo, OpenAI-compatible
API, part of the [Operium Orchestra](https://operium.ru) stack.*

---

## Возможности / Features

- **OpenAI-совместимый API** на порту `:8100` — `/v1/chat/completions`, стриминг через SSE
- **Расширенное API моделей** — `/api/models`, `/api/models/{id}/load`, `/api/system`
- **Импорт моделей из LM Studio** — `POST /api/models/import` принимает путь и подхватывает .gguf без копирования
- **Бэкенды llama.cpp** — Metal (Apple Silicon), CUDA (NVIDIA), BLAS / чистый CPU
- **Reasoning-модели** — корректно обрабатывает `<think>` блоки Qwen/DeepSeek
- **Hot-swap моделей** — загрузка/выгрузка без перезапуска процесса

---

## Сборка / Build

```bash
# Apple Silicon (Metal, GPU-ускорение)
make build-metal

# NVIDIA (CUDA)
make build-cuda

# Без GPU (CPU, для серверов и CI)
make build-cpu
```

Бинарник появится в `bin/orchestra-runtime`. Для первой сборки нужны:
- Go ≥ 1.22
- cmake ≥ 3.22
- (Metal) Xcode Command Line Tools
- (CUDA) CUDA Toolkit ≥ 12

Первая сборка компилит llama.cpp (~5 мин на M-серии, ~15 мин на CPU).
Последующие пересборки — секунды.

---

## Запуск / Run

```bash
./bin/orchestra-runtime
# Listening on :8100
```

По умолчанию сервер слушает `:8100`. Настройки через переменные окружения:

| Переменная | По умолчанию | Что делает |
|------------|-------------|------------|
| `ORCHESTRA_PORT` | `8100` | Порт HTTP-сервера |
| `ORCHESTRA_MODELS_DIR` | `~/.orchestra/models` | Каталог с .gguf |
| `ORCHESTRA_LOG_LEVEL` | `info` | `debug` / `info` / `warn` |
| `ORCHESTRA_CTX_SIZE` | `4096` | Явный runtime-wide `n_ctx`; planner не уменьшает заданное значение |
| `ORCHESTRA_AUTO_FIT` | `1` | Автоматически подбирает batch, KV cache и context при нехватке памяти |
| `ORCHESTRA_ALLOW_MEMORY_OVERCOMMIT` | `0` | Отключает preflight RAM budget для ручной диагностики |

---

## API

### Список моделей

```bash
curl :8100/api/models
```

```jsonc
[
  {
    "id": "qwen2.5-coder-7b-q4",
    "name": "Qwen2.5 Coder 7B",
    "parameters": "7B",
    "quantization": "Q4_K_M",
    "size_human": "4.7 GB",
    "status": "ready"   // или "loaded" / "loading" / "error"
  }
]
```

### Загрузка модели в память

```bash
curl -X POST :8100/api/models/qwen2.5-coder-7b-q4/load \
  -H 'Content-Type: application/json' \
  -d '{"gpu_layers": -1, "context_size": 8192}'
```

`gpu_layers: -1` = все слои на GPU. `context_size` — n_ctx.

При загрузке без тела runtime строит безопасный профиль автоматически. Если
backend действительно вернул OOM, загрузка повторяется с последовательно более
лёгкими `n_batch`, KV cache и context-профилями. Явно переданные
`context_size`, `batch_size`, `kv_cache_quant_k` и `kv_cache_quant_v` planner не
меняет. Для полностью ручного режима передайте `"auto_fit": false`.

### Чат (OpenAI-совместимо)

```bash
curl -X POST :8100/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen2.5 Coder 7B",
    "stream": true,
    "messages": [
      {"role": "user", "content": "Напиши quicksort на Go"}
    ]
  }'
```

SSE-стрим с `data: {...}` чанками + `data: [DONE]` в конце.

### Импорт из LM Studio

```bash
curl -X POST :8100/api/models/import \
  -H 'Content-Type: application/json' \
  -d '{"path": "~/.lmstudio/models"}'
```

Рекурсивно сканирует каталог на .gguf и регистрирует их в каталоге моделей
**без копирования** — просто добавляет ссылки.

### Состояние сервера

```bash
curl :8100/api/system
```

```jsonc
{
  "engine_state": "ready",   // "ready" / "loading" / "busy"
  "current_model": "qwen2.5-coder-7b-q4",
  "backend": "metal",
  "memory_used_mb": 4823
}
```

---

## Архитектура / Architecture

```
cmd/
  orchestra-runtime/main.go    ← entry-point
internal/
  server/                      ← chi router, middleware, SSE
  handler/                     ← REST-handlers (models.go, chat.go, system.go)
  service/                     ← бизнес-логика (model_manager, chat_runner)
  repository/                  ← storage layer (модели на диске)
  llama/                       ← CGo-binding к llama.cpp
llama.cpp/                     ← git submodule (upstream)
```

Слоистая архитектура: `handler → service → repository`. Всё GPU-общение
изолировано в `internal/llama`, который компилируется с нужными
`-DGGML_METAL/CUDA` флагами.

---

## Клиенты / Clients

- [`operium-orchestra-vscode`](https://github.com/operium/operium-orchestra-vscode) — расширение VS Code, автостарт и UI
- [`orchestra-desktop`](../orchestra-desktop) — нативный Tauri-клиент

---

## Release channels

Расширение **Operium Orchestra** по умолчанию тянет канал `stable` с GitHub Raw:

`https://raw.githubusercontent.com/IlyaAbramychev/orchestra-runtime/main/release-manifests/stable.json`  
(переопределяется настройкой `orchestra.runtimeManifestBaseUrl`).

- В этом репозитории поддерживается [`release-manifests/stable.json`](./release-manifests/stable.json): в нём перечислены прямые URL на бинарники в **GitHub Releases** (`…/releases/download/v…/orchestra-runtime-…`).
- Пока для указанного тега **нет опубликованного release с нужными именами файлов**, автоустановка из VS Code завершится ошибкой скачивания — сначала нужно прогнать релизный CI (`.github/workflows/release.yml` после тега `v*`) и прикрепить артефакты, либо выставить свой CDN и перегенерировать манифест (`scripts/generate-runtime-manifest.mjs`, см. workflow `runtime-channel-manifest.yml`).

Дополнительно: `operium-orchestra-vscode/docs/runtime-update-manifest.md` и `docs/runtime-release-channels.md` в репозитории расширения.

---

## Зависимости / Third-Party

- [llama.cpp](https://github.com/ggerganov/llama.cpp) (MIT) — движок инференса GGUF
- [chi](https://github.com/go-chi/chi) (MIT) — HTTP-роутинг

Полный список в [NOTICE](./NOTICE).

---

## License

© 2026 Operium. All rights reserved. See [LICENSE](./LICENSE).
