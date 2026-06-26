# Chat tools & structured output — model matrix

Qualification status for OpenAI-style chat extensions via Nadir Gateway → mlx-lm / mlx-vlm.

!!! warning "Living document"
    Update this table when you validate a new MLX weight on Apple Silicon. Gateway behaviour is identical for all rows — differences are **upstream only**.

Last updated: June 2026.

## Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Validated or natively supported by mlx-lm with correct `tokenizer_config.json` |
| ⚠️ | Relay works; model may ignore tools or produce loose JSON |
| ❌ | Not supported / wrong launch mode |

## Gateway (all TEXT / MULTIMODAL aliases)

| Feature | Gateway |
|---------|---------|
| `tools` + `tool_choice` relay | ✅ deep-copy, no truncation |
| `response_format` relay | ✅ |
| Upstream error passthrough | ✅ HTTP status + JSON body |
| SSE `stream: true` with tools | ✅ relay |

## TEXT models (mlx-lm)

| Model family | Tools | `json_object` | `json_schema` | Notes |
|--------------|-------|---------------|---------------|-------|
| **Qwen3-Coder** (`mlx-community/Qwen3-Coder-*`) | ✅ | ⚠️ | ⚠️ | Requires `tool_parser_type: qwen3_coder` in `tokenizer_config.json` |
| **Qwen3** instruct (tool_parser in config) | ✅ | ⚠️ | ⚠️ | Depends on mlx-community `tokenizer_config.json` |
| **MiniMax M2** variants | ✅ | ⚠️ | ⚠️ | `tool_parser_type: minimax_m2` |
| **Qwen2.5** instruct | ⚠️ | ⚠️ | ⚠️ | Chat works; tools often ignored without parser metadata |
| **Gemma 3 / 4** text & VLM | ⚠️ | ⚠️ | ⚠️ | Use for chat/VLM; tool calling not primary focus |
| **Llama 3.x** instruct | ⚠️ | ⚠️ | ⚠️ | Prompt-only JSON; no native tool parser in most MLX builds |

## MULTIMODAL (mlx-vlm)

| Feature | Status | Notes |
|---------|--------|-------|
| `tools` on `/v1/chat/completions` | ⚠️ | Follow mlx-vlm server capabilities; gateway relays |
| Vision + tools | ⚠️ | Validate per model in UI before production agent use |

## Structured output reality (mlx-lm v1)

| OpenAI field | mlx-lm server |
|--------------|---------------|
| `response_format.type: json_object` | Relayed; **not enforced** by server |
| `response_format.type: json_schema` | Relayed; **no schema validation** |

**Workaround:** instruct the model explicitly (“respond with valid JSON only”) and validate client-side.

## How to check tool support on a running instance

```bash
# Direct instance (replace port)
curl -s http://127.0.0.1:11400/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @- <<'EOF'
{
  "model": "default_model",
  "messages": [{"role": "user", "content": "Call ping"}],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "ping",
        "parameters": {"type": "object", "properties": {}}
      }
    }
  ]
}
EOF
```

Inspect `choices[0].message.tool_calls` and instance logs for mlx-lm warnings.

## References

- [gateway-runbooks/chat.md](gateway-runbooks/chat.md)
- mlx-lm: [PR #217](https://github.com/ml-explore/mlx-lm/pull/217)
