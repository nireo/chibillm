"""Model-free tests for the structured benchmark reader and session driver."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location(
    'benchmark', Path(__file__).resolve().parents[1] / 'scripts' / 'benchmark.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class BenchmarkTests(unittest.TestCase):
    def test_hashes_only_answer_bytes_including_embedded_delimiters(self):
        answer = '€\nyou> qwen> trailing space '.encode()
        stdout = b'banner\nyou> qwen> ' + answer + b'\nyou> history cleared\nyou> qwen> \n'
        rows = [{'output_bytes': len(answer)}, {'output_bytes': 0}]
        self.assertEqual(benchmark.answer_hashes(stdout, rows), [
            hashlib.sha256(answer).hexdigest(), hashlib.sha256(b'').hexdigest()])
        with self.assertRaises(RuntimeError):
            benchmark.answer_hashes(stdout, [{'output_bytes': len(answer) - 1}])

    def test_validates_schema_count_and_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'metrics.jsonl'
            row = {'schema_version': 1, 'type': 'request', 'status': 'ok'}
            path.write_text(json.dumps(row) + '\n')
            self.assertEqual(benchmark.read_records(path, 1), [row])
            with self.assertRaises(RuntimeError):
                benchmark.read_records(path, 2)
            for field, value in [('schema_version', 2), ('status', 'error')]:
                path.write_text(json.dumps({**row, field: value}))
                with self.assertRaises(RuntimeError):
                    benchmark.read_records(path, 1)

    def test_warm_session_resets_history_without_restarting_process(self):
        args = SimpleNamespace(binary='chibillm', model='model', context_length=4096,
                               stream=False, timeout=10)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'metrics.jsonl'
            row = {'schema_version': 1, 'type': 'request', 'status': 'ok', 'output_bytes': 1}

            def execute(command, **kwargs):
                self.assertIn('--no-stream', command)
                self.assertIn('--progress=off', command)
                self.assertEqual(kwargs['input'], b'hello\n/reset\n' * 4 + b'/quit\n')
                path.write_text((json.dumps(row) + '\n') * 4)
                return subprocess.CompletedProcess(command, 0, stdout=b'qwen> A\n' * 4, stderr=b'')

            with patch.object(benchmark.subprocess, 'run', side_effect=execute) as run:
                rows, process = benchmark.run_session(args, 'hello', 32, 4, path)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(row['answer_sha256'] == hashlib.sha256(b'A').hexdigest() for row in rows))
            self.assertGreaterEqual(process['process_seconds'], 0)

    def test_summaries_skip_missing_rates(self):
        keys = ['tokenization_seconds', 'prefill_seconds', 'engine_ttft_seconds',
                'first_text_seconds', 'decode_engine_seconds', 'text_decode_seconds',
                'output_write_seconds', 'end_to_end_seconds',
                'prefill_tokens_per_second', 'decode_tokens_per_second']
        rows = [{key: None for key in keys} for _ in range(3)]
        for row, value in zip(rows, [1, 3, 2]):
            row['end_to_end_seconds'] = value
        summary = benchmark.summarize(rows)
        self.assertIsNone(summary['decode_tokens_per_second'])
        self.assertEqual(summary['end_to_end_seconds'], {'samples': 3, 'median': 2, 'min': 1, 'max': 3})


if __name__ == '__main__':
    unittest.main()
