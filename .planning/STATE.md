---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: agentic-reliability
status: executing
stopped_at: Phase 1 implemented and measured
last_updated: "2026-09-14T13:05:00+03:00"
last_activity: 2026-09-14
progress:
  total_phases: 6
  completed_phases: 1
  percent: 17
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-09-06)

**Core value:** Operium Desktop runs local agents on Orchestra Runtime with
Ollama/LM Studio-level reliability: responsive while busy, no crashes on long
context, streaming with tools, cheap follow-up turns.
**Current focus:** Phase 2 — Устойчивость исполнения

## Current Position

Phase: 1 (Неблокирующий control plane и наблюдаемость) — IMPLEMENTED, awaiting commit/review
Branch: `gsd/phase-1-nonblocking-control-plane`
Next: Phase 2, items 1–2 (context guard before prefill, subprocess by default)

Progress: [██░░░░░░░░] 17%

## Phase 1 — результат

### Что сделано

| Требование | Изменение |
|---|---|
| RESP-01, RESP-02 | `Engine` публикует неизменяемый снимок состояния в `atomic.Pointer`; `State/IsLoaded/LoadedModelID/LoadedContextSize/LoadedOptions/LoadedAt/LastError/ModelDesc/IdleTimeout` больше не берут `e.mu`. Обработчик `status` воркера автоматически стал неблокирующим. |
| RESP-01 | Сэмплирование железа (`sysctl`, `vm_stat`) вынесено с пути запроса в фоновое обновление; кэш GGUF-метаданных прогревается при старте. |
| RESP-03 | `InitBackend` идёт в горутине и в сервере, и в воркере; `LoadModel` ждёт готовности бэкенда; `/health` отдаёт `backend: starting|ready`, `/api/system` и `/api/status` — `starting`. |
| RESP-04 | `engine_state` во время генерации — `generating` (раньше `ready`). |
| RESP-05 | Снят `WriteTimeout` (10 мин) с HTTP-сервера; добавлен `ReadHeaderTimeout`. |
| OPS-01 | Логи llama.cpp/ggml/mtmd идут через неблокирующий pipe (`log_bridge.c`) в slog с `component=llama.cpp`; INFO llama.cpp понижен до debug, WARN/ERROR сохраняют уровень. |

### Замеры (M4 Pro 24 GB, Qwen3.5-9B Q4, тот же метод, что в CONCERNS.md)

| Метрика | До (v0.4.0, bed95d8) | После (in-process) | После (subprocess) |
|---|---|---|---|
| Порт отвечает после старта | 6.85 c | 0.10 c (TCP 0.06 c) | 0.03 c |
| `/api/tags` во время генерации | 9.37 c | ≤ 1.0 мс | ≤ 1.0 мс |
| `/api/system` во время загрузки | 0.91 c | ≤ 3.7 мс (57 проб) | 0.3 мс |
| `engine_state` во время генерации | `ready` | `generating` | `generating` |
| Не-JSON строк в логе | 1189 из 1226 | 0 из 40 | 0 из 45 |

Тесты: `go vet ./...`, `go test ./...`, `go test -race` для `engine`, `service`,
`supervisor` — зелёные. Добавлены тесты: неблокирующие читатели статуса,
ожидание бэкенда в `LoadModel`, состояние `starting`, фоновое обновление
железа, декодер нативных логов (разбиение строк, CONT-записи, ограничение
длины).

### Известные остатки

- `/api/ps` один раз из 18 проб ответил за 0.32 c во время генерации. Это не
  блокировка: `ModelManager.Get` на каждый запрос выполняет
  `normalizeModelMetadata` (поиск mmproj через `ReadDir` и `os.Stat`). Нужен
  кэш нормализованных записей — перенесено в Phase 2.
- Критерий «SSE-стрим 15 минут не обрывается» проверен по построению (таймаут
  записи удалён), но не прогоном на 15 минут.
- Первый запрос `/api/system` до завершения фонового сэмплирования железа всё
  ещё ждёт его (один раз за жизнь процесса).

## Проверка Operium Progress Protocol (2026-09-14)

Метод: ответы рантайма (Qwen3.5-9B, системный промпт Desktop, запрос как в
`runtimeClient.chatStream`) прогнаны через **настоящие** функции Desktop
`extractExecutions → extractUsage → extractProgress → splitReasoning`,
вырезанные из `OrchestraDesktop/src/hooks/use-chat.js` (порядок как в
`ChatView.jsx:905-907`, панель прогресса — `ChatView.jsx:3075`).

| Сценарий | До исправления | После |
|---|---|---|
| Без tools, `max_tokens` 4096 | Работает: 3 блока, статусы `in_progress → done`, утечек нет | Без изменений: 4 блока, утечек нет |
| С tools (агентный режим) | **Сломано**: текст и блоки прогресса выбрасывались, панель пуста | Панель заполняется (`done, done, pending`), текст виден |
| Без `max_tokens` | Обрыв на 512 токенах посреди размышлений | Не менялось; Desktop сам шлёт ≥ 2048 |

### Найдено и исправлено в рантайме

- `handleOpenAIToolStream` и нестримовый `/v1` отбрасывали `content`, если
  модель вызывала инструмент. Теперь текст отдаётся рядом с нативными
  вызовами, размышления идут первыми. Вызовы, восстановленные из сырого текста
  (совместимость со старыми воркерами), по-прежнему не дублируются в `content`.
- То же на `/api/chat`: нестримовый ответ обнулял `content`, буферизованный
  стрим терял `thinking` и `content`.
- Тесты: 4 новых (OpenAI и Ollama, стрим и нестрим).

### Требует правки в Operium Desktop (патч: `handoff/operium-desktop-reasoning.diff`)

1. `runtimeClient.js` читает только `delta.content`. В режиме tools рантайм
   отдаёт размышления в `reasoning_content`, поэтому область размышлений пуста в
   каждом раунде с инструментами. С патчем: 0 → 901 символ.
2. `splitReasoning` берёт только первый `<think>`. В многораундовом ответе
   размышления последующих раундов уходили в ответ вместе с тегами. С патчем
   утечек нет.
3. Фраза системного промпта «Для reasoning-моделей выводи размышления в
   <think>...</think>» заставляет Qwen3.5 цитировать тег внутри размышлений (6
   пар тегов в одном ответе), что обрывает область размышлений. Её стоит убрать:
   модели с нативным thinking сами выводят `<think>`.

### Полный агентный цикл (3 раунда, как в Desktop)

- Раунды 1–2: текст виден (212 и 54 символа), вызовы `write_file` проходят.
- Ни в одном раунде модель не вывела блок прогресса (в одиночном запросе C —
  вывела два). На 9B-модели в режиме tools протокол соблюдается ненадёжно.
- Раунд 3: модель зациклилась, генерируя содержимое теста, упёрлась в
  `max_tokens` посреди `<tool_call>` и получила `tool_protocol_error`. Рантайм
  в этом случае отдаёт сырой текст в `content`, и Desktop показывает в ответе
  10 КБ с `</think>` и XML вызова. Кандидат в фазу 2: при ошибке протокола не
  отдавать синтаксис вызова как текст ответа и сообщать причину.

### Остаётся (фаза 3)

- В режиме tools ответ по-прежнему буферизуется до конца генерации: TTFB 32.8 c
  на раунде из ~1000 токенов. Прогресс и текст приходят, но разом.
- Инструкция прогресса добавляется в первое system-сообщение, до AGENTS.md и
  навыка; модель могла бы следовать ей надёжнее в конце промпта.

## Decisions

- `/health` сохраняет `"status":"ok"`; новое поле `backend` добавлено, чтобы не
  сломать существующих клиентов, проверяющих `status`.
- Логи llama.cpp пишутся в pipe, а не через cgo-колбэк: ggml логирует из
  Metal-потоков и из деструкторов при выходе процесса, где вызов в Go может
  зависнуть или упасть.
- `LoadedModelID()` во время загрузки возвращает загружаемую модель; все
  потребители дополнительно проверяют `IsLoaded()`.

## Pending Todos

- Решение по коммиту: в ветке, помимо фазы 1, лежат незакоммиченные изменения
  `operium_progress` (`internal/handler/chat.go`, `chat_test.go`,
  `handler.go`, `internal/model/chat.go`), которые к фазе 1 не относятся.
