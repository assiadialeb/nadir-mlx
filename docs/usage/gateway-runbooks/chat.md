# Runbook — Gateway CHAT (tools & JSON)

Validate `POST /v1/chat/completions` through Nadir Gateway (`:11380`) for a **RUNNING** TEXT or MULTIMODAL instance.

!!! note "Gateway vs backend"
    The gateway **relays** `tools`, `tool_choice`, and `response_format` unchanged. Tool parsing and JSON enforcement happen in **mlx-lm** / **mlx-vlm** on the instance port.

## Prerequisites

- Django + gateway running
- TEXT instance **Running** with a known alias (tool-capable models: see [chat-tools-model-matrix.md](../chat-tools-model-matrix.md))
- `mlx-lm` recent enough for server tool calling (2025-06+)

## 1. Discovery

```bash
curl -s http://127.0.0.1:11380/v1/models | python3 -m json.tool
```

## 2. Basic chat

```bash
curl -s http://127.0.0.1:11380/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<alias>",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}]
  }' | python3 -m json.tool
```

**Expected:** HTTP 200, `choices[0].message.content` non-empty.

## 3. Tool calling

Use a model with `tool_parser_type` in `tokenizer_config.json` (e.g. Qwen3-Coder family).

```bash
curl -s http://127.0.0.1:11380/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<tool-capable-alias>",
    "messages": [
      {"role": "user", "content": "What is the weather in Paris?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get weather for a city",
          "parameters": {
            "type": "object",
            "properties": {
              "city": {"type": "string"}
            },
            "required": ["city"]
          }
        }
      }
    ],
    "tool_choice": "auto"
  }' | python3 -m json.tool
```

**Expected (tool-capable model):** HTTP 200, `choices[0].message.tool_calls` with `function.name` and JSON `arguments`.

**Expected (model without tool support):** HTTP 200 with plain text, or mlx-lm warning in instance logs — tools ignored upstream.

## 4. JSON object mode (relay)

```bash
curl -s http://127.0.0.1:11380/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<alias>",
    "messages": [
      {"role": "user", "content": "Return only JSON with keys city and country for Paris."}
    ],
    "response_format": {"type": "json_object"}
  }' | python3 -m json.tool
```

**Expected:** Gateway forwards `response_format`. Valid JSON in `message.content` depends on the model — validate with `python3 -m json.tool` on the content string.

## 5. JSON schema mode (relay)

```bash
curl -s http://127.0.0.1:11380/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<alias>",
    "messages": [
      {"role": "user", "content": "Describe Paris briefly."}
    ],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "city_fact",
        "strict": true,
        "schema": {
          "type": "object",
          "properties": {
            "city": {"type": "string"},
            "fact": {"type": "string"}
          },
          "required": ["city", "fact"],
          "additionalProperties": false
        }
      }
    }
  }' | python3 -m json.tool
```

**Expected:** Payload reaches mlx-lm unchanged. Schema compliance is **not** guaranteed by mlx-lm v1 — treat as best-effort.

## 6. Streaming with tools

```bash
curl -N http://127.0.0.1:11380/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<tool-capable-alias>",
    "stream": true,
    "messages": [{"role": "user", "content": "Ping"}],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "ping",
          "parameters": {"type": "object", "properties": {}}
        }
      }
    ]
  }'
```

**Expected:** `text/event-stream` chunks; tool call fragments depend on mlx-lm streaming support.

## Troubleshooting

| HTTP | Cause | Action |
|------|--------|--------|
| 404 | Unknown alias | Check `GET /v1/models` |
| 503 | Instance stopped | Start TEXT server in UI |
| 400 | Wrong route (e.g. EMBEDDING alias) | Use `/v1/chat/completions` only |
| 422 / 500 | Upstream mlx-lm error | Inspect instance logs; gateway relays body |
| 200 but no `tool_calls` | Model lacks `tool_parser_type` | See model matrix; pick tool-capable weights |
| 200 but invalid JSON | mlx-lm ignores `response_format` | Strengthen prompt or pick instruction-tuned model |

## References

- [chat-tools-model-matrix.md](../chat-tools-model-matrix.md)
