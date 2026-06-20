#!/usr/bin/env python3
"""
LLM OpenAI 相容 API 效能基準測試工具
=====================================
針對任何 OpenAI 相容的 LLM API 伺服器，測量延遲、吞吐量與並發擴展性。
支援 Ollama、MLX-LM、vLLM、LM Studio、llama.cpp server。

使用方式：
    python llmbench.py
    python llmbench.py --host 192.168.1.100 --port 8001
    python llmbench.py --concurrency 1 4 8 16 --output results.json

    # Ollama（預設，連接埠 11434）
    python llmbench.py --model gemma4:e4b

    # MLX-LM（連接埠 8080）
    python llmbench.py --host localhost --port 8080 --model mlx-community/gemma-4-e4b-it-4bit

    # vLLM（連接埠 8001）
    python llmbench.py --host localhost --port 8001

系統需求：
    pip install httpx rich
"""

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Optional

try:
    import httpx
except ImportError:
    print("缺少相依套件：pip install httpx")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.table import Table
    from rich import print as rprint
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# ---------------------------------------------------------------------------
# 測試提示詞 — 短 / 中 / 長
# 每個類別包含 5 個提示詞，測試時循環使用
# ---------------------------------------------------------------------------
PROMPTS = {
    "short": [
        "What is 2 + 2?",
        "Name the capital of France.",
        "What color is the sky?",
        "Who wrote Hamlet?",
        "What is H2O?",
    ],
    "medium": [
        "Explain the attention mechanism in transformer models in 3 sentences.",
        "What are the key differences between supervised and unsupervised learning?",
        "Summarize the main causes of World War I.",
        "Describe how gradient descent works in neural network training.",
        "What is the CAP theorem in distributed systems?",
    ],
    "long": [
        "Write a detailed explanation of how convolutional neural networks work, including convolution layers, pooling, activation functions, and fully connected layers.",
        "Explain the complete lifecycle of an HTTP request from the moment a user types a URL in the browser until the page is rendered.",
        "Describe the differences between LSTM, GRU, and Transformer architectures for sequence modeling tasks, including their advantages and limitations.",
        "Write a comprehensive overview of reinforcement learning, covering Markov decision processes, policy gradient methods, Q-learning, and actor-critic approaches.",
        "Explain how large language models are trained, covering pre-training, fine-tuning, RLHF, and the role of tokenization.",
    ],
}

# 各類別最大生成 Token 數
MAX_TOKENS = {
    "short": 20,    # 短提示詞：最多 20 tokens
    "medium": 100,  # 中提示詞：最多 100 tokens
    "long": 400,    # 長提示詞：最多 400 tokens
}


# ---------------------------------------------------------------------------
# 資料類別
# ---------------------------------------------------------------------------
@dataclass
class RequestResult:
    prompt_tokens: int       # 輸入提示詞的 token 數
    completion_tokens: int   # 模型生成的 token 數
    ttft_ms: float           # 首個 token 時間，毫秒（串流模式才有）
    total_ms: float          # 完整請求總耗時，毫秒
    success: bool            # 請求是否成功
    error: Optional[str] = None  # 失敗時的錯誤訊息

    @property
    def tokens_per_sec(self) -> float:
        """計算每秒生成 token 數（單一請求）"""
        if self.total_ms <= 0:
            return 0.0
        return self.completion_tokens / (self.total_ms / 1000)


@dataclass
class BenchmarkResult:
    scenario: str           # 情境名稱，例如 medium_conc4
    concurrency: int        # 並發請求數
    num_requests: int       # 總請求數
    prompt_category: str    # 提示詞類別：short / medium / long
    results: list = field(default_factory=list, repr=False)

    @property
    def successes(self):
        """取得所有成功的請求結果"""
        return [r for r in self.results if r.success]

    def summary(self) -> dict:
        """彙整統計數據：延遲百分位數、吞吐量等"""
        ok = self.successes
        if not ok:
            return {"error": "all requests failed"}
        total_ms_list = [r.total_ms for r in ok]
        tps_list      = [r.tokens_per_sec for r in ok]
        ttft_list     = [r.ttft_ms for r in ok if r.ttft_ms > 0]

        # 以最慢請求的完成時間作為牆鐘時間估算（近似值）
        elapsed_wall = max(r.total_ms for r in ok) / 1000
        total_tokens = sum(r.completion_tokens for r in ok)

        return {
            "scenario":           self.scenario,
            "concurrency":        self.concurrency,
            "num_requests":       self.num_requests,
            "prompt_category":    self.prompt_category,
            "success_rate":       f"{len(ok)}/{self.num_requests}",
            "latency_p50_ms":     round(statistics.median(total_ms_list), 1),   # 中位數延遲
            "latency_p95_ms":     round(sorted(total_ms_list)[int(len(total_ms_list) * 0.95)], 1),  # P95 延遲
            "latency_p99_ms":     round(sorted(total_ms_list)[int(len(total_ms_list) * 0.99)], 1),  # P99 延遲
            "ttft_p50_ms":        round(statistics.median(ttft_list), 1) if ttft_list else "N/A",   # 首個 token 中位數時間
            "tps_per_req_median": round(statistics.median(tps_list), 1),        # 單一請求中位數 tok/s
            "total_tokens_out":   total_tokens,                                  # 總輸出 token 數
            "aggregate_tps":      round(total_tokens / elapsed_wall, 1) if elapsed_wall > 0 else 0,  # 總吞吐量 tok/s
        }


# ---------------------------------------------------------------------------
# 核心基準測試邏輯
# ---------------------------------------------------------------------------
async def _stream_chat_completion(
    client: httpx.AsyncClient,
    base_url: str,
    payload: dict[str, object],
    t0: float,
) -> tuple[int, int, float]:
    prompt_tokens = 0
    completion_tokens = 0
    ttft_ms = 0.0
    async with client.stream(
        "POST", f"{base_url}/v1/chat/completions",
        json=payload, timeout=120,
    ) as resp:
        resp.raise_for_status()
        first_chunk = True
        async for line in resp.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            if first_chunk:
                ttft_ms = (time.perf_counter() - t0) * 1000
                first_chunk = False
            chunk = json.loads(line[6:])
            if chunk.get("choices"):
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    completion_tokens += 1
            usage = chunk.get("usage") or {}
            if usage.get("completion_tokens"):
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                completion_tokens = usage["completion_tokens"]
    return prompt_tokens, completion_tokens, ttft_ms


async def _blocking_chat_completion(
    client: httpx.AsyncClient,
    base_url: str,
    payload: dict[str, object],
) -> tuple[int, int]:
    resp = await client.post(
        f"{base_url}/v1/chat/completions",
        json=payload, timeout=120,
    )
    resp.raise_for_status()
    usage = resp.json().get("usage", {})
    return usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


async def single_request(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    stream: bool,
) -> RequestResult:
    """
    發送單一請求並量測效能指標。
    串流模式：逐行解析 SSE 事件，記錄首個 token 時間（TTFT）。
    非串流模式：等待完整回應後一次取得 usage 統計。
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
        # include_usage=True：要求 Ollama/MLX-LM 在串流結束時回傳正確的 token 計數
        "stream_options": {"include_usage": True} if stream else None,
    }

    t0 = time.perf_counter()
    ttft_ms = 0.0

    try:
        if stream:
            prompt_tokens, completion_tokens, ttft_ms = await _stream_chat_completion(
                client, base_url, payload, t0,
            )
        else:
            prompt_tokens, completion_tokens = await _blocking_chat_completion(
                client, base_url, payload,
            )

        total_ms = (time.perf_counter() - t0) * 1000
        return RequestResult(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            success=True,
        )

    except Exception as e:
        # 請求失敗：記錄錯誤但不中斷整體測試
        total_ms = (time.perf_counter() - t0) * 1000
        return RequestResult(
            prompt_tokens=0, completion_tokens=0,
            ttft_ms=0, total_ms=total_ms,
            success=False, error=str(e),
        )


async def run_concurrent(
    base_url: str,
    model: str,
    prompts: list[str],
    max_tokens: int,
    concurrency: int,
    num_requests: int,
    temperature: float,
    stream: bool,
) -> list[RequestResult]:
    """
    使用 Semaphore 控制並發數，同時發送 num_requests 個請求。
    提示詞循環使用（i % len(prompts)），確保測試多樣性。
    """
    semaphore = asyncio.Semaphore(concurrency)  # 限制最大並發數
    async with httpx.AsyncClient() as client:
        async def bounded(prompt):
            async with semaphore:
                return await single_request(
                    client, base_url, model, prompt, max_tokens, temperature, stream
                )

        tasks = [
            bounded(prompts[i % len(prompts)])
            for i in range(num_requests)
        ]
        results = await asyncio.gather(*tasks)  # 並發執行所有任務
    return list(results)


# ---------------------------------------------------------------------------
# 自動從伺服器偵測模型名稱
# ---------------------------------------------------------------------------
def get_model(base_url: str, override: Optional[str]) -> str:
    """
    若未指定 --model，自動呼叫 /v1/models 取得第一個可用模型 ID。
    適用於 Ollama、MLX-LM、vLLM 等相容伺服器。
    """
    if override:
        return override
    try:
        resp = httpx.get(f"{base_url}/v1/models", timeout=10)
        models = resp.json()["data"]
        if models:
            return models[0]["id"]
    except Exception:
        pass
    raise RuntimeError(f"無法從 {base_url}/v1/models 自動偵測模型，請使用 --model 指定。")


# ---------------------------------------------------------------------------
# 結果顯示
# ---------------------------------------------------------------------------
def print_summary(summaries: list[dict]):
    """
    以表格形式輸出所有情境的彙整結果。
    已安裝 rich 套件時顯示彩色表格，否則輸出 TSV 格式。
    """
    if HAS_RICH:
        console = Console()
        table = Table(title="LLM 基準測試結果", show_lines=True)
        cols = [
            "scenario", "concurrency", "prompt_category",
            "success_rate", "latency_p50_ms", "latency_p95_ms",
            "ttft_p50_ms", "tps_per_req_median", "aggregate_tps",
        ]
        for c in cols:
            table.add_column(c, style="cyan", no_wrap=True)
        for s in summaries:
            table.add_row(*[str(s.get(c, "")) for c in cols])
        console.print(table)
    else:
        # 無 rich 時退回純文字 TSV 輸出
        header = ["scenario", "concurrency", "prompt_category",
                  "success_rate", "p50_ms", "p95_ms", "ttft_ms", "tok/s/req", "agg_tok/s"]
        print("\t".join(header))
        for s in summaries:
            row = [
                s.get("scenario", ""), str(s.get("concurrency", "")),
                s.get("prompt_category", ""), s.get("success_rate", ""),
                str(s.get("latency_p50_ms", "")), str(s.get("latency_p95_ms", "")),
                str(s.get("ttft_p50_ms", "")), str(s.get("tps_per_req_median", "")),
                str(s.get("aggregate_tps", "")),
            ]
            print("\t".join(row))


# ---------------------------------------------------------------------------
# 指令列參數解析
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="LLM OpenAI API 效能基準測試工具")

    # 伺服器連線設定
    # 環境變數 OLLAMA_BENCH_HOST / OLLAMA_BENCH_PORT 可覆蓋預設值
    parser.add_argument("--host",         default=os.getenv("OLLAMA_BENCH_HOST", "localhost"),
                        help="伺服器主機名稱或 IP（預設：localhost）")
    parser.add_argument("--port",         type=int, default=int(os.getenv("OLLAMA_BENCH_PORT", "11434")),
                        help="伺服器連接埠（預設：11434 for Ollama，8080 for MLX-LM，8001 for vLLM）")

    # 模型設定
    parser.add_argument("--model",        default=os.getenv("MODEL_ID"),
                        help="模型 ID，未指定時自動從伺服器偵測")

    # 測試情境設定
    parser.add_argument("--concurrency",  type=int, nargs="+", default=[1, 4, 8, 16],
                        help="並發數掃描清單，例如 --concurrency 1 2 4 8 16")
    parser.add_argument("--categories",   nargs="+", default=["short", "medium", "long"],
                        choices=["short", "medium", "long"],
                        help="提示詞類別：short（20 tokens）/ medium（100 tokens）/ long（400 tokens）")
    parser.add_argument("--num-requests", type=int, default=20,
                        help="每個情境的請求總數（建議 20-50，越多結果越穩定）")
    parser.add_argument("--temperature",  type=float, default=0.0,
                        help="採樣溫度，0.0 為確定性輸出（方便結果重現）")

    # 串流設定
    parser.add_argument("--stream",       action="store_true", default=True,
                        help="啟用 SSE 串流（預設開啟，啟用後可量測 TTFT）")
    parser.add_argument("--no-stream",    dest="stream", action="store_false",
                        help="停用串流，改用非串流模式（較慢，無 TTFT 量測）")

    # 輸出設定
    parser.add_argument("--output",       default=None,
                        help="將完整結果儲存為 JSON 檔案，例如 --output results.json")

    return parser.parse_args()


def _write_benchmark_output(path: str, meta: dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)


async def main():
    args = parse_args()
    base_url = f"http://{args.host}:{args.port}"

    print(f"\n{'='*60}")
    print(f"  LLM 基準測試  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  端點     : {base_url}")

    # 自動偵測或使用指定的模型 ID
    model = get_model(base_url, args.model)
    print(f"  模型     : {model}")
    print(f"  情境     : 並發數={args.concurrency}, 類別={args.categories}")
    print(f"  請求數   : 每情境 {args.num_requests} 個請求")
    print(f"  串流模式 : {args.stream}")
    print(f"{'='*60}\n")

    all_summaries = []
    all_raw = []

    # 遍歷所有提示詞類別與並發數組合
    for category in args.categories:
        prompts  = PROMPTS[category]
        max_toks = MAX_TOKENS[category]

        for conc in args.concurrency:
            tag = f"[{category}/conc={conc}]"
            print(f"  執行中 {tag} ...")
            sys.stdout.flush()

            t_wall = time.perf_counter()
            results = await run_concurrent(
                base_url, model, prompts, max_toks,
                conc, args.num_requests, args.temperature, args.stream,
            )
            wall_sec = time.perf_counter() - t_wall

            br = BenchmarkResult(
                scenario=f"{category}_conc{conc}",
                concurrency=conc,
                num_requests=args.num_requests,
                prompt_category=category,
                results=results,
            )
            summ = br.summary()
            summ["wall_sec"] = round(wall_sec, 2)  # 實際牆鐘時間（秒）
            all_summaries.append(summ)

            ok = len(br.successes)
            print(f"    完成: {ok}/{args.num_requests} 成功 | "
                  f"p50={summ.get('latency_p50_ms')}ms | "
                  f"總吞吐={summ.get('aggregate_tps')} tok/s")

            # 若指定輸出檔案，儲存每筆請求的原始數據
            if args.output:
                all_raw.append({
                    "scenario": br.scenario,
                    "summary": summ,
                    "requests": [asdict(r) for r in results],
                })

    print()
    print_summary(all_summaries)

    # 將完整結果（含每筆請求原始數據）寫入 JSON
    if args.output:
        meta = {
            "timestamp": datetime.now().isoformat(),
            "endpoint":  base_url,
            "model":     model,
            "config": {
                "concurrency":  args.concurrency,
                "categories":   args.categories,
                "num_requests": args.num_requests,
                "temperature":  args.temperature,
                "stream":       args.stream,
            },
            "results": all_raw,
        }
        await asyncio.to_thread(_write_benchmark_output, args.output, meta)
        print(f"\n  結果已儲存 → {args.output}")

    print()


if __name__ == "__main__":
    asyncio.run(main())
