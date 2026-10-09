project(
  'chibillm',
  'cpp',
  'objcpp',
  version: '0.1.0',
  default_options: [
    'cpp_std=c++23',
    'warning_level=3',
  ],
)

add_project_arguments('-fobjc-arc', language: 'objcpp')

src_include = include_directories('src')

python = import('python').find_installation('python3')
fs = import('fs')
metal_source_parts = []
foreach name : ['common.metalh', 'linear.metal', 'tensor_ops.metal', 'sampling.metal', 'attention.metal', 'deltanet.metal']
  # fs.read tracks each source so edits automatically regenerate the header.
  contents = fs.read('src/metal/kernels' / name)
  contents = contents.replace('#include "common.metalh"\n', '\n')
  if contents.contains(')CHIBILLM_METAL"')
    error('Metal source contains the C++ raw-string delimiter')
  endif
  metal_source_parts += 'R"CHIBILLM_METAL(' + '#line 1 "' + name + '"\n' + contents + '\n)CHIBILLM_METAL"'
endforeach
kernel_config = configuration_data()
kernel_config.set('METAL_KERNEL_SOURCE', ',\n'.join(metal_source_parts))
kernel_source = configure_file(
  input: 'src/metal/metal_kernel_source.h.in',
  output: 'metal_kernel_source.h',
  configuration: kernel_config,
)

metal_framework_dep = dependency(
  'appleframeworks',
  modules: [
    'Foundation',
    'Metal',
  ],
)

doctest_dep = dependency(
  'doctest',
  fallback: ['doctest', 'doctest_dep'],
)

nlohmann_json_dep = dependency(
  'nlohmann_json',
  fallback: ['nlohmann_json', 'nlohmann_json_dep'],
)

cpp_httplib_dep = dependency(
  'cpp-httplib',
  fallback: ['cpp-httplib', 'cpp_httplib_dep'],
)

chibillm_model_format = static_library(
  'chibillm_model_format',
  'src/model_format/safetensors.cc',
  include_directories: src_include,
  dependencies: nlohmann_json_dep,
  install: false,
)

chibillm_core = static_library(
  'chibillm_core',
  [
    'src/seq.cc',
    'src/prefix_cache.cc',
    'src/block_manager.cc',
    'src/scheduler.cc',
    'src/model_batch.cc',
    'src/model_runner.cc',
    'src/serving_runtime.cc',
    'src/inference_engine.cc',
  ],
  include_directories: src_include,
  install: false,
)

chibillm_metal = static_library(
  'chibillm_metal',
  [
    'src/metal/metal_context.mm',
    'src/metal/metal_kernels.mm',
    'src/metal/metal_kernels_linear.mm',
    'src/metal/metal_kernels_attention.mm',
    'src/metal/metal_kernels_deltanet.mm',
    'src/tensor/deltanet.cc',
    'src/tensor/attention.cc',
    'src/metal/metal_kv_cache.cc',
    'src/model_format/weight_reader.cc',
    'src/tensor/tensor_ops.cc',
    'src/tensor/functions.cc',
    'src/tensor/matrix.cc',
  ],
  include_directories: src_include,
  dependencies: metal_framework_dep,
  link_with: chibillm_model_format,
  install: false,
)

chibillm_qwen = static_library(
  'chibillm_qwen',
  [
    'src/qwen/qwen_configs.cc',
    'src/qwen/qwen3_5_model_state.cc',
    'src/qwen/qwen_layer.cc',
    'src/qwen/qwen_model_runner.cc',
    'src/qwen/qwen_tokenizer.cc',
    'src/qwen/qwen_weights.cc',
  ],
  include_directories: src_include,
  dependencies: nlohmann_json_dep,
  link_with: [
    chibillm_core,
    chibillm_model_format,
    chibillm_metal,
  ],
  install: false,
)

chibillm_models = static_library(
  'chibillm_models', 'src/model_factory.cc',
  include_directories: src_include,
  dependencies: nlohmann_json_dep,
  link_with: chibillm_qwen,
  install: false,
)

chibillm_server = static_library(
  'chibillm_server',
  'src/server.cc',
  include_directories: src_include,
  dependencies: [
    cpp_httplib_dep,
    nlohmann_json_dep,
  ],
  link_with: chibillm_core,
  install: false,
)

executable(
  'chibillm',
  [
    'src/main.cpp',
    'src/cli_options.cc',
    'src/repl.cc',
    'src/generation_metrics.cc',
    kernel_source,
  ],
  include_directories: src_include,
  dependencies: [
    cpp_httplib_dep,
    nlohmann_json_dep,
    metal_framework_dep,
  ],
  link_with: [
    chibillm_models,
    chibillm_server,
  ],
  install: true,
)

test('chibillm benchmark tests', python, args: [meson.project_source_root() / 'tests/benchmark_test.py'])

test_args = [
  '-DQWEN3_CONFIG_FIXTURE_PATH="' + meson.project_source_root() / 'tests/fixtures/qwen3_0_6b_config.json' + '"',
  '-DQWEN3_5_CONFIG_FIXTURE_PATH="' + meson.project_source_root() / 'tests/fixtures/qwen3_5_0_8b_config.json' + '"',
  '-DQWEN3_5_MODEL_PATH="' + meson.project_source_root() / 'qwen3_5_model' + '"',
]

foreach spec : [
  {'name': 'unit', 'sources': ['tests/seq_test.cc', 'tests/block_manager_test.cc'], 'library': chibillm_core},
  {'name': 'repl', 'sources': ['tests/repl_test.cc', 'tests/generation_metrics_test.cc', 'src/repl.cc', 'src/cli_options.cc', 'src/generation_metrics.cc'], 'library': chibillm_core, 'deps': [nlohmann_json_dep]},
  {'name': 'scheduler', 'sources': ['tests/scheduler_test.cc'], 'library': chibillm_core},
  {'name': 'prefix cache', 'sources': ['tests/prefix_cache_test.cc'], 'library': chibillm_core},
  {'name': 'model batch', 'sources': ['tests/model_batch_test.cc'], 'library': chibillm_core},
  {'name': 'inference engine', 'sources': ['tests/inference_engine_test.cc', 'tests/fake_model_runner.cc'], 'library': chibillm_core},
  {'name': 'server', 'sources': ['tests/server_test.cc', 'tests/serving_runtime_test.cc'], 'library': chibillm_server, 'deps': [cpp_httplib_dep, nlohmann_json_dep]},
  {'name': 'qwen hybrid state', 'sources': ['tests/qwen3_5_model_state_test.cc'], 'library': chibillm_qwen},
  {'name': 'qwen hybrid layers', 'sources': ['tests/qwen3_5_layer_test.cc'], 'library': chibillm_qwen},
  {'name': 'qwen config', 'sources': ['tests/qwen_config_test.cc'], 'library': chibillm_qwen, 'deps': [nlohmann_json_dep]},
  {'name': 'qwen tokenizer', 'sources': ['tests/qwen_tokenizer_test.cc'], 'library': chibillm_qwen, 'deps': [nlohmann_json_dep]},
  {'name': 'safetensors', 'sources': ['tests/safetensors_test.cc'], 'library': chibillm_model_format, 'deps': [nlohmann_json_dep]},
  {'name': 'qwen weights', 'sources': ['tests/qwen_weights_test.cc'], 'library': chibillm_models, 'deps': [nlohmann_json_dep]},
  {'name': 'prompt cache generation', 'sources': ['tests/prompt_cache_generation_test.cc'], 'library': chibillm_models},
  {'name': 'kv cache', 'sources': ['tests/metal_kv_cache_test.cc'], 'library': chibillm_metal},
  {'name': 'tensor', 'sources': ['tests/tensor_test.cc'], 'library': chibillm_metal},
  {'name': 'deltanet', 'sources': ['tests/deltanet_test.cc'], 'library': chibillm_metal},
  {'name': 'tensor ops', 'sources': ['tests/tensor_ops_test.cc'], 'library': chibillm_metal},
  {'name': 'quantization', 'sources': ['tests/quantization_test.cc'], 'library': chibillm_metal},
]
  name = spec['name'] == 'unit' ? 'chibillm_tests' : 'chibillm_' + spec['name'].replace(' ', '_') + '_tests'
  target = executable(
    name, spec['sources'] + [kernel_source],
    include_directories: [src_include, include_directories('tests')],
    dependencies: [doctest_dep] + spec.get('deps', []),
    cpp_args: test_args,
    link_with: spec['library'],
  )
  test('chibillm ' + spec['name'] + ' tests', target)
endforeach
