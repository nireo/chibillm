#!/usr/bin/env python3
"""Benchmark full-precision request metrics: same-process warm or fresh-process cold runs."""
import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

PROMPTS = [
    'Explain in three sentences why the sky is blue.',
    ' '.join(['A small inference engine batches requests, stores attention state in paged memory, '
              'and executes tensor operations on a GPU.'] * 16)
    + ' Summarize this description in three sentences.',
]


def positive(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return value


def read_records(path, expected):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if len(rows) != expected:
        raise RuntimeError(f'expected {expected} request records, got {len(rows)}')
    for row in rows:
        if row.get('schema_version') != 1 or row.get('type') != 'request':
            raise RuntimeError('unsupported metrics schema')
        if row.get('status') != 'ok':
            raise RuntimeError(f'request failed at {row.get("stage")}: {row.get("stop_reason")}')
    return rows


def answer_hashes(stdout, rows):
    """Use exact UTF-8 byte lengths, not delimiters that may appear inside an answer."""
    cursor = 0
    hashes = []
    for row in rows:
        start = stdout.find(b'qwen> ', cursor)
        if start < 0:
            raise RuntimeError('missing answer prefix')
        start += len(b'qwen> ')
        size = row['output_bytes']
        if not isinstance(size, int) or size < 0:
            raise RuntimeError('invalid output byte count')
        end = start + size
        if stdout[end:end + 1] != b'\n':
            raise RuntimeError('answer framing does not match metrics byte count')
        answer = stdout[start:end]
        answer.decode('utf-8')  # Reject incomplete output rather than hashing it as success.
        hashes.append(hashlib.sha256(answer).hexdigest())
        cursor = end + 1
    return hashes


def run_session(args, prompt, budget, count, metrics_path):
    command = [args.binary, '--context-length', str(args.context_length),
               '--max-tokens', str(budget), '--progress=off',
               '--metrics-jsonl', str(metrics_path)]
    if not args.stream:
        command.append('--no-stream')
    command.append(args.model)
    # Each request starts fresh, while warm mode keeps the loaded model and pools alive.
    request_input = ((prompt + '\n/reset\n') * count + '/quit\n').encode()
    started = time.perf_counter()
    run = subprocess.run(command, input=request_input, capture_output=True,
                         check=True, timeout=args.timeout)
    process_seconds = time.perf_counter() - started
    try:
        rows = read_records(metrics_path, count)
        hashes = answer_hashes(run.stdout, rows)
    except (RuntimeError, ValueError) as error:
        raise RuntimeError(f'{error}\n{run.stderr.decode(errors="replace")}') from error
    for row, digest in zip(rows, hashes):
        row['answer_sha256'] = digest
    return rows, {'command': command, 'process_seconds': process_seconds}


def summarize(rows):
    keys = ['tokenization_seconds', 'prefill_seconds', 'engine_ttft_seconds',
            'first_text_seconds', 'decode_engine_seconds', 'text_decode_seconds',
            'output_write_seconds', 'end_to_end_seconds',
            'prefill_tokens_per_second', 'decode_tokens_per_second']
    result = {}
    for key in keys:
        values = [row[key] for row in rows if row[key] is not None]
        result[key] = ({'samples': len(values), 'median': statistics.median(values),
                        'min': min(values), 'max': max(values)} if values else None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', default='build/chibillm')
    parser.add_argument('--model', default='qwen3_5_model')
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=['warm', 'cold', 'both'], default='warm')
    parser.add_argument('--iterations', type=positive, default=3)
    parser.add_argument('--max-tokens', type=positive, nargs='+', default=[32, 128])
    parser.add_argument('--context-length', type=positive, default=4096)
    parser.add_argument('--timeout', type=positive, default=300)
    parser.add_argument('--stream', action='store_true', help='include incremental output costs')
    args = parser.parse_args()
    if args.context_length % 16:
        parser.error('--context-length must be a multiple of 16')

    with open(args.binary, 'rb') as binary:
        binary_hash = hashlib.file_digest(binary, 'sha256').hexdigest()
    cpu = platform.processor()
    if platform.system() == 'Darwin':
        cpu = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
    report = {'schema_version': 1, 'environment': {
        'platform': platform.platform(), 'machine': platform.machine(), 'cpu': cpu,
        'binary_sha256': binary_hash, 'model_directory': str(Path(args.model).resolve()),
    }, 'processes': [], 'results': [], 'summaries': []}
    modes = ['cold', 'warm'] if args.mode == 'both' else [args.mode]
    with tempfile.TemporaryDirectory(prefix='chibillm-benchmark-') as directory:
        for prompt_index, prompt in enumerate(PROMPTS):
            for budget in args.max_tokens:
                case = f'prompt-{prompt_index}-output-{budget}'
                for mode in modes:
                    # Cold: each measured request loads a fresh process. OS caches may be warm.
                    # Warm: one unmeasured request followed by measured requests in that process.
                    sessions = args.iterations if mode == 'cold' else 1
                    for _ in range(sessions):
                        process_id = len(report['processes'])
                        count = 1 if mode == 'cold' else args.iterations + 1
                        rows, process = run_session(args, prompt, budget, count,
                            Path(directory) / f'{process_id}.jsonl')
                        process.update(process_id=process_id, mode=mode, case=case)
                        report['processes'].append(process)
                        for iteration, row in enumerate(rows):
                            row.update(process_id=process_id, case=case, mode=mode,
                                       iteration=iteration if mode == 'warm' else _,
                                       warmup=mode == 'warm' and iteration == 0,
                                       prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
                            report['results'].append(row)
                            print(json.dumps(row), flush=True)
                    measured = [row for row in report['results']
                                if row['case'] == case and row['mode'] == mode and not row['warmup']]
                    summary = {'case': case, 'mode': mode, 'metrics': summarize(measured)}
                    report['summaries'].append(summary)
                    print(json.dumps(summary), flush=True)
    # Preserve raw runs even if determinism fails, so the discrepancy is inspectable.
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    for case in {row['case'] for row in report['results']}:
        hashes = {row['answer_sha256'] for row in report['results'] if row['case'] == case}
        if len(hashes) != 1:
            raise RuntimeError(f'output mismatch for {case}; inspect {args.output}')


if __name__ == '__main__':
    main()
