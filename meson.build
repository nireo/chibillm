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
    'src/block_manager.cc',
    'src/scheduler.cc',
    'src/model_batch.cc',
    'src/model_runner.cc',
    'src/text_decoder.cc',
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
    'src/tensor/output.cc',
    'src/tensor/embedding.cc',
    'src/tensor/layers.cc',
    'src/tensor/attention.cc',
    'src/metal/metal_kv_cache.cc',
    'src/model_format/weight_reader.cc',
    'src/tensor/tensor_ops.cc',
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
    'src/qwen/qwen_chat.cc',
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
  'src/main.cpp',
  include_directories: src_include,
  cpp_args: [
    '-DCHIBILLM_SHADER_PATH="' + join_paths(
      meson.project_source_root(),
      'shaders',
      'tensor_ops.metal',
    ) + '"',
  ],
  dependencies: [
    cpp_httplib_dep,
    metal_framework_dep,
  ],
  link_with: [
    chibillm_models,
    chibillm_server,
  ],
  install: true,
)

test_args = [
  '-DCHIBILLM_SHADER_PATH="' + meson.project_source_root() / 'shaders/tensor_ops.metal' + '"',
  '-DQWEN3_CONFIG_FIXTURE_PATH="' + meson.project_source_root() / 'tests/fixtures/qwen3_0_6b_config.json' + '"',
  '-DQWEN3_5_CONFIG_FIXTURE_PATH="' + meson.project_source_root() / 'tests/fixtures/qwen3_5_0_8b_config.json' + '"',
  '-DQWEN3_5_MODEL_PATH="' + meson.project_source_root() / 'qwen3_5_model' + '"',
]

foreach spec : [
  {'name': 'unit', 'sources': ['tests/seq_test.cc', 'tests/block_manager_test.cc'], 'library': chibillm_core},
  {'name': 'scheduler', 'sources': ['tests/scheduler_test.cc'], 'library': chibillm_core},
  {'name': 'model batch', 'sources': ['tests/model_batch_test.cc'], 'library': chibillm_core},
  {'name': 'inference engine', 'sources': ['tests/inference_engine_test.cc', 'tests/fake_model_runner.cc'], 'library': chibillm_core},
  {'name': 'server', 'sources': ['tests/server_test.cc'], 'library': chibillm_server, 'deps': [cpp_httplib_dep, nlohmann_json_dep]},
  {'name': 'qwen config', 'sources': ['tests/qwen_config_test.cc'], 'library': chibillm_qwen, 'deps': [nlohmann_json_dep]},
  {'name': 'qwen tokenizer', 'sources': ['tests/qwen_tokenizer_test.cc'], 'library': chibillm_qwen, 'deps': [nlohmann_json_dep]},
  {'name': 'safetensors', 'sources': ['tests/safetensors_test.cc'], 'library': chibillm_model_format, 'deps': [nlohmann_json_dep]},
  {'name': 'qwen weights', 'sources': ['tests/qwen_weights_test.cc'], 'library': chibillm_qwen, 'deps': [nlohmann_json_dep]},
  {'name': 'kv cache', 'sources': ['tests/metal_kv_cache_test.cc'], 'library': chibillm_metal},
  {'name': 'tensor', 'sources': ['tests/tensor_test.cc'], 'library': chibillm_metal},
  {'name': 'tensor ops', 'sources': ['tests/tensor_ops_test.cc'], 'library': chibillm_metal},
]
  name = spec['name'] == 'unit' ? 'chibillm_tests' : 'chibillm_' + spec['name'].replace(' ', '_') + '_tests'
  target = executable(
    name, spec['sources'],
    include_directories: [src_include, include_directories('tests')],
    dependencies: [doctest_dep] + spec.get('deps', []),
    cpp_args: test_args,
    link_with: spec['library'],
  )
  test('chibillm ' + spec['name'] + ' tests', target)
endforeach
