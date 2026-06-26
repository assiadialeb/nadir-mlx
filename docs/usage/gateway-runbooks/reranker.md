# Runbook — Gateway RERANKER (MLX-27)

Validate `POST /v1/rerank` through Nadir Gateway (`:11380`) for a **RUNNING** reranker instance.

## Prerequisites

- Django + gateway running
- RERANKER instance **Running** (JinaForRanking mlx-community or local-reranker compatible model)
- Gateway alias configured (e.g. `jina-reranker-v3`)

!!! note "Required model field"
    The gateway routes via the JSON `model` field (your alias). Even when the upstream reranker ignores it, **`model` is required** on the gateway.

## 1. Discovery

```bash
curl -s http://127.0.0.1:11380/v1/models | python3 -m json.tool
```

Confirm alias with `"launch_mode": "RERANKER"`.

## 2. Rerank request

```bash
curl -s http://127.0.0.1:11380/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<alias>",
    "query": "What is Python?",
    "documents": [
      "Python is a programming language.",
      "The weather is sunny today.",
      "Python supports machine learning libraries."
    ],
    "top_n": 2,
    "return_documents": true
  }' | python3 -m json.tool
```

**Expected:** HTTP 200, `results` sorted by `relevance_score` descending; Python-related docs rank higher.

## 3. Empty documents

```bash
curl -s http://127.0.0.1:11380/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<alias>",
    "query": "test",
    "documents": []
  }' | python3 -m json.tool
```

**Expected:** HTTP 200, `"results": []`.

## 4. Wrong route (negative test)

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:11380/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<alias>",
    "messages": [{"role": "user", "content": "Hi"}]
  }'
```

**Expected:** HTTP **400**.

## Troubleshooting

| HTTP | Cause | Action |
|------|--------|--------|
| 400 | Missing `model` | Always pass gateway alias in `model` |
| 404 | Unknown alias | Verify alias in UI |
| 503 | Instance stopped | Start reranker in UI |
| 500 upstream | Model incompatible | Use JinaForRanking or supported local-reranker checkpoint |
