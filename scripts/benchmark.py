#!/usr/bin/env python3
"""Fixed short/long prompt benchmark; one warmup and three measured runs."""
import argparse
import hashlib
import json
import re
import statistics
import subprocess
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--binary', default='build/chibillm')
parser.add_argument('--model', default='qwen_model')
parser.add_argument('--output', required=True)
args = parser.parse_args()
prompts = [
    'Explain in three sentences why the sky is blue.',
    ' '.join(['A small inference engine batches requests, stores attention state in paged memory, '
              'and executes tensor operations on a GPU.'] * 16)
    + ' Summarize this description in three sentences.',
]
pattern = re.compile(
    r'\[perf\] prompt (\d+) tok \| output (\d+) tok \| first ([\d.]+) s '
    r'\| prefill ([\d.]+) tok/s \| decode ([\d.]+) tok/s '
    r'\(p50 ([\d.]+) ms, p95 ([\d.]+) ms\) \| total ([\d.]+) s'
)
results = []
for iteration in range(4):
    for case, prompt in enumerate(prompts):
        started = time.perf_counter()
        run = subprocess.run([args.binary, args.model], input=prompt + '\n/quit\n',
                             text=True, capture_output=True, check=True)
        process_seconds = time.perf_counter() - started
        match = pattern.search(run.stderr)
        if not match:
            raise RuntimeError(run.stderr)
        row = dict(zip(['prompt', 'output', 'ttft', 'prefill', 'decode', 'p50', 'p95', 'total'],
                       map(float, match.groups())))
        row.update(case=case, iteration=iteration, process_seconds=process_seconds,
                   hash=hashlib.sha256(run.stdout.encode()).hexdigest())
        results.append(row)
        print(json.dumps(row), flush=True)
Path(args.output).write_text(json.dumps(results, indent=2) + '\n')
for case in range(len(prompts)):
    runs = [row for row in results if row['case'] == case and row['iteration'] > 0]
    medians = {key: statistics.median(row[key] for row in runs)
               for key in ['prefill', 'decode', 'p50', 'p95', 'total']}
    print(f'case {case} measured medians: {json.dumps(medians)}')
