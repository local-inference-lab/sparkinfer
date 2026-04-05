"""CLI for b12x serve — interactive, one-shot, and API server modes.

Usage:
    python -m serve.cli /path/to/model --chat
    python -m serve.cli /path/to/model --tp 2 --gpu-ids 4,5 --chat
    python -m serve.cli /path/to/model --prompt "Hello" --temperature 0.7
    python -m serve.cli /path/to/model --serve --port 8000
"""

from __future__ import annotations

import sys
import time

from serve.runtime_warnings import import_torch_safely

torch = import_torch_safely()
torch.set_grad_enabled(False)

from serve.logging import configure_logging, get_logger
from serve.engine.sampling import SamplingParams
from serve.engine.serving import ServingEngine
from serve.tp.launch import launch_tp

LOGGER = get_logger(__name__)


def _parse_graph_batch_sizes(spec: str | None, *, enabled: bool) -> list[int]:
    if not enabled:
        return []
    if spec is None:
        return [1, 2, 4, 8]
    values = [int(value) for value in spec.split(",") if value.strip()]
    if not values:
        return []
    return sorted(set(values))


def _run(tp_group, model_path, max_tokens, chat, prompt_text, temperature, top_p, top_k, rep_penalty,
         serve_mode=False, port=8000, capture_prefill_graph=False, enable_thinking=True, no_graph=False,
         compile_layers=True, load_backend="auto", log_level="info", graph_batch_sizes=None,
         prefill_chunk_size=512, max_running=None, max_waiting=256, max_prefill_tokens=4096,
         enable_webui=False):
    rank = tp_group.rank if tp_group else 0
    device = f"cuda:{tp_group.device.index}" if tp_group else "cuda"
    configure_logging(log_level, rank=rank)

    graph_sizes = _parse_graph_batch_sizes(graph_batch_sizes, enabled=not no_graph)
    engine = ServingEngine(model_path, device=device, tp_group=tp_group,
                           load_backend=load_backend,
                           graph_batch_sizes=graph_sizes,
                           prefill_chunk_size=prefill_chunk_size,
                           capture_prefill_graph=capture_prefill_graph,
                           compile_layers=compile_layers,
                           max_running=max_running,
                           max_waiting=max_waiting,
                           max_prefill_tokens=max_prefill_tokens)

    if rank != 0:
        engine.run_follower()
        return

    if serve_mode:
        from serve.api.server import ServingApp
        app = ServingApp(engine, enable_webui=enable_webui)
        LOGGER.info(f"Starting API server on port {port}")
        try:
            app.run(port=port)
        finally:
            engine.shutdown()
        return

    # Reasoning models need sampling, not greedy. Use model-appropriate defaults
    # when the user hasn't explicitly set temperature.
    if chat and temperature == 0.0:
        temperature = 0.6
        if top_p == 1.0:
            top_p = 0.95
        if top_k == -1:
            top_k = 20

    params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=rep_penalty,
        max_new_tokens=max_tokens,
    )

    if prompt_text:
        # One-shot: warmup then generate.
        engine.complete("warmup", SamplingParams.greedy(max_new_tokens=3))

        t0 = time.time()
        if chat:
            result = engine.chat([
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_text},
            ], params)
        else:
            result = engine.complete(prompt_text, params)
        elapsed = time.time() - t0

        text = engine.tokenizer.decode(result.generated_ids, skip_special_tokens=True)
        n = len(result.generated_ids)
        print(text, flush=True)
        print(f"[{n} tokens in {elapsed:.1f}s, {n/elapsed:.1f} tok/s, TTFT={result.time_to_first_token_ms:.0f}ms]",
              file=sys.stderr, flush=True)
        engine.shutdown()
        return

    # Interactive mode.
    tokenizer = engine.tokenizer
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    print()
    while True:
        try:
            prompt = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt.strip():
            continue

        t0 = time.time()
        if chat:
            messages.append({"role": "user", "content": prompt})
            template_kwargs = {"add_generation_prompt": True}
            if not enable_thinking:
                template_kwargs["enable_thinking"] = False
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, **template_kwargs)
            input_ids = tokenizer.encode(formatted, add_special_tokens=False)
        else:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()

        # Stream tokens to stdout.
        n = 0
        result = None
        gen_text_parts = []
        for _tok_id, tok_text, result in engine.generate_stream(input_ids, params):
            print(tok_text, end="", flush=True)
            gen_text_parts.append(tok_text)
            n += 1
        elapsed = time.time() - t0

        # Accumulate assistant response for multi-turn.
        if chat:
            full_response = "".join(gen_text_parts)
            # Separate thinking from content for proper template round-tripping.
            msg = {"role": "assistant"}
            if "</think>" in full_response:
                think_part, content_part = full_response.split("</think>", 1)
                think_part = think_part.replace("<think>", "").strip()
                msg["reasoning_content"] = think_part
                msg["content"] = content_part.strip()
            else:
                msg["content"] = full_response
            messages.append(msg)

        ttft = result.time_to_first_token_ms if result else 0
        print(f"\n\n[{n} tokens in {elapsed:.1f}s, {n/elapsed:.1f} tok/s, TTFT={ttft:.0f}ms]")
        print()

    engine.shutdown()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="b12x serve CLI")
    parser.add_argument("model_path", help="Path to HF model")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel degree (1-8)")
    parser.add_argument("--gpu-ids", type=str, default=None, help="Comma-separated GPU IDs")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--chat", action="store_true", help="Use chat template")
    parser.add_argument("--prompt", type=str, default=None, help="One-shot prompt")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0=greedy)")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling threshold")
    parser.add_argument("--top-k", type=int, default=-1, help="Top-K sampling (-1=disabled)")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Repetition penalty")
    parser.add_argument("--serve", action="store_true", help="Start HTTP API server")
    parser.add_argument("--web-ui", action="store_true",
                        help="Enable the built-in browser chat UI at /ui (server mode only)")
    parser.add_argument("--port", type=int, default=8000, help="API server port")
    parser.set_defaults(capture_prefill_graph=False)
    parser.add_argument("--capture-prefill-graph", dest="capture_prefill_graph", action="store_true",
                        help="Enable prefill CUDA graph capture")
    parser.add_argument("--no-prefill-graph", dest="capture_prefill_graph", action="store_false",
                        help=argparse.SUPPRESS)
    parser.add_argument(
        "--graph-batch-sizes",
        type=str,
        default=None,
        help="Comma-separated CUDA graph batch sizes (default: 1,2,4,8)",
    )
    parser.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=512,
        help="Chunk size for long-prompt prefill scheduling and graph capture",
    )
    parser.add_argument(
        "--max-running",
        type=int,
        default=None,
        help="Maximum concurrent running+prefilling requests",
    )
    parser.add_argument(
        "--max-waiting",
        type=int,
        default=256,
        help="Maximum queued requests before submit() rejects new work",
    )
    parser.add_argument(
        "--max-prefill-tokens",
        type=int,
        default=4096,
        help="Maximum batched prompt tokens admitted in one prefill step",
    )
    parser.add_argument("--no-think", action="store_true",
                        help="Disable thinking/reasoning for reasoning models (Qwen3.5)")
    parser.add_argument("--no-graph", action="store_true",
                        help="Disable CUDA graph capture (run eager)")
    parser.set_defaults(compile_layers=True)
    parser.add_argument("--compile", dest="compile_layers", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--no-compile", dest="compile_layers", action="store_false",
                        help="Disable per-layer torch.compile")
    parser.add_argument(
        "--load-backend",
        choices=("auto", "distributed", "local"),
        default="auto",
        help="Model loading backend",
    )
    parser.add_argument(
        "--log-level",
        choices=("debug", "info", "warning", "error"),
        default="info",
        help="Serve log verbosity",
    )
    args = parser.parse_args()

    gpu_ids = None
    if args.gpu_ids:
        gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
        assert len(gpu_ids) == args.tp

    launch_tp(
        _run,
        world_size=args.tp,
        args=(args.model_path, args.max_tokens, args.chat, args.prompt,
              args.temperature, args.top_p, args.top_k, args.repetition_penalty,
              args.serve, args.port, args.capture_prefill_graph,
              not args.no_think, args.no_graph, args.compile_layers, args.load_backend, args.log_level,
              args.graph_batch_sizes, args.prefill_chunk_size, args.max_running,
              args.max_waiting, args.max_prefill_tokens, args.web_ui),
        gpu_ids=gpu_ids,
    )


if __name__ == "__main__":
    main()
