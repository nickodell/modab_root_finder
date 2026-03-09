py.extension_module(
  'original_paper',
  'original_paper.pyx',
  dependencies: py_dep,
  override_options : ['cython_language=cpp'],
  install: true,
  include_directories: [],
  subdir: 'modab_root_finder'
)

py.extension_module(
  'modern',
  'modern.pyx',
  dependencies: py_dep,
  override_options : ['cython_language=cpp'],
  install: true,
  include_directories: [],
  subdir: 'modab_root_finder'
)

py.extension_module(
  'step_introspect',
  'step_introspect.pyx',
  dependencies: py_dep,
  override_options : ['cython_language=cpp'],
  install: true,
  include_directories: [],
  subdir: 'modab_root_finder'
)

python_sources = [
  '__init__.py',
  'modab_from_proektsoftbg.py',
  'mpmath_root.py',
]

py.install_sources(
  python_sources,
  subdir: 'modab_root_finder'
)

install_subdir('tests', install_dir: py.get_install_dir() / 'modab_root_finder')
