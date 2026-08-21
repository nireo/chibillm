project(
  'chibillm',
  'cpp',
  version: '0.1.0',
  default_options: [
    'cpp_std=c++23',
    'warning_level=3',
  ],
)

src_include = include_directories('src')

doctest_dep = dependency(
  'doctest',
  fallback: ['doctest', 'doctest_dep'],
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
