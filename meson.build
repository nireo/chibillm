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
  'src/seq.cc',
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
  'tests/seq_test.cc',
  include_directories: src_include,
  dependencies: doctest_dep,
  link_with: chibillm_core,
)

test('chibillm unit tests', chibillm_tests)
