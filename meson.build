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
    'src/inference_engine.cc',
  ],
  include_directories: src_include,
  install: false,
)

chibillm_metal = static_library(
  'chibillm_metal',
  [
    'src/metal/metal_context.mm',
    'src/metal/metal_kv_cache.cc',
    'src/tensor/tensor_ops.cc',
  ],
  include_directories: src_include,
  dependencies: metal_framework_dep,
  install: false,
)

chibillm_qwen = static_library(
  'chibillm_qwen',
  [
    'src/qwen/qwen_config.cc',
    'src/qwen/qwen_embedding.cc',
    'src/qwen/qwen_layer.cc',
    'src/qwen/qwen_output.cc',
    'src/qwen/qwen_weights.cc',
  ],
  include_directories: src_include,
  dependencies: nlohmann_json_dep,
  link_with: [
    chibillm_model_format,
    chibillm_metal,
  ],
  install: false,
)

executable(
  'chibillm',
  'src/main.cpp',
  include_directories: src_include,
  link_with: chibillm_core,
  install: true,
)

chibillm_tests = executable(
  'chibillm_tests',
  [
    'tests/seq_test.cc',
    'tests/block_manager_test.cc',
  ],
  include_directories: src_include,
  dependencies: doctest_dep,
  link_with: chibillm_core,
)

test('chibillm unit tests', chibillm_tests)

chibillm_scheduler_tests = executable(
  'chibillm_scheduler_tests',
  'tests/scheduler_test.cc',
  include_directories: src_include,
  dependencies: doctest_dep,
  link_with: chibillm_core,
)

test('chibillm scheduler tests', chibillm_scheduler_tests)

chibillm_model_batch_tests = executable(
  'chibillm_model_batch_tests',
  'tests/model_batch_test.cc',
  include_directories: src_include,
  dependencies: doctest_dep,
  link_with: chibillm_core,
)

test('chibillm model batch tests', chibillm_model_batch_tests)

chibillm_inference_engine_tests = executable(
  'chibillm_inference_engine_tests',
  'tests/inference_engine_test.cc',
  include_directories: src_include,
  dependencies: doctest_dep,
  link_with: chibillm_core,
)

test('chibillm inference engine tests', chibillm_inference_engine_tests)

qwen_config_fixture_path = join_paths(
  meson.project_source_root(),
  'tests',
  'fixtures',
  'qwen3_0_6b_config.json',
)

chibillm_qwen_config_tests = executable(
  'chibillm_qwen_config_tests',
  'tests/qwen_config_test.cc',
  include_directories: src_include,
  cpp_args: [
    '-DQWEN_CONFIG_FIXTURE_PATH="' + qwen_config_fixture_path + '"',
  ],
  dependencies: [
    doctest_dep,
    nlohmann_json_dep,
  ],
  link_with: chibillm_qwen,
)

test('chibillm qwen config tests', chibillm_qwen_config_tests)

chibillm_safetensors_tests = executable(
  'chibillm_safetensors_tests',
  'tests/safetensors_test.cc',
  include_directories: [
    src_include,
    include_directories('tests'),
  ],
  dependencies: [
    doctest_dep,
    nlohmann_json_dep,
  ],
  link_with: chibillm_model_format,
)

test('chibillm safetensors tests', chibillm_safetensors_tests)

metal_shader_path = join_paths(
  meson.project_source_root(),
  'shaders',
  'tensor_ops.metal',
)

chibillm_qwen_weights_tests = executable(
  'chibillm_qwen_weights_tests',
  'tests/qwen_weights_test.cc',
  include_directories: [
    src_include,
    include_directories('tests'),
  ],
  cpp_args: [
    '-DCHIBILLM_SHADER_PATH="' + metal_shader_path + '"',
  ],
  dependencies: [
    doctest_dep,
    metal_framework_dep,
    nlohmann_json_dep,
  ],
  link_with: [
    chibillm_model_format,
    chibillm_metal,
    chibillm_qwen,
  ],
)

test('chibillm qwen weights tests', chibillm_qwen_weights_tests)

chibillm_metal_tests = executable(
  'chibillm_metal_tests',
  'tests/metal_test.cc',
  include_directories: src_include,
  cpp_args: [
    '-DCHIBILLM_SHADER_PATH="' + metal_shader_path + '"',
  ],
  dependencies: [
    doctest_dep,
    metal_framework_dep,
  ],
  link_args: ['-Wl,-no_warn_duplicate_libraries'],
  link_with: [
    chibillm_core,
    chibillm_metal,
  ],
)

test('chibillm metal tests', chibillm_metal_tests)

chibillm_kv_cache_tests = executable(
  'chibillm_kv_cache_tests',
  'tests/metal_kv_cache_test.cc',
  include_directories: src_include,
  cpp_args: [
    '-DCHIBILLM_SHADER_PATH="' + metal_shader_path + '"',
  ],
  dependencies: [
    doctest_dep,
    metal_framework_dep,
  ],
  link_args: ['-Wl,-no_warn_duplicate_libraries'],
  link_with: [
    chibillm_core,
    chibillm_metal,
  ],
)

test('chibillm kv cache tests', chibillm_kv_cache_tests)

chibillm_tensor_tests = executable(
  'chibillm_tensor_tests',
  'tests/tensor_test.cc',
  include_directories: src_include,
  cpp_args: [
    '-DCHIBILLM_SHADER_PATH="' + metal_shader_path + '"',
  ],
  dependencies: [
    doctest_dep,
    metal_framework_dep,
  ],
  link_args: ['-Wl,-no_warn_duplicate_libraries'],
  link_with: [
    chibillm_core,
    chibillm_metal,
  ],
)

test('chibillm tensor tests', chibillm_tensor_tests)

chibillm_tensor_ops_tests = executable(
  'chibillm_tensor_ops_tests',
  'tests/tensor_ops_test.cc',
  include_directories: src_include,
  cpp_args: [
    '-DCHIBILLM_SHADER_PATH="' + metal_shader_path + '"',
  ],
  dependencies: [
    doctest_dep,
    metal_framework_dep,
  ],
  link_args: ['-Wl,-no_warn_duplicate_libraries'],
  link_with: [
    chibillm_core,
    chibillm_metal,
  ],
)

test('chibillm tensor ops tests', chibillm_tensor_ops_tests)
