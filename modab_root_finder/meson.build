py.extension_module(
  '_core',
  'coremodule.c',
  install: true,
  subdir: 'modab_root_finder'
)

py.extension_module(
    'rect',
    'rect.pyx',
    dependencies: py_dep,
    override_options : ['cython_language=cpp'],
    install: true,
    include_directories: [],
)

py.extension_module(
    'modab_from_paper',
    'modab_from_paper.pyx',
    dependencies: py_dep,
    override_options : ['cython_language=cpp'],
    install: true,
    include_directories: [],
)

python_sources = [
  '__init__.py',
  'conftest.py',
  'modab_from_proektsoftbg_encoded.py',
  'mpmath_root.py',
]

py.install_sources(
  python_sources,
  subdir: 'modab_root_finder'
)

install_subdir('submodule', install_dir: py.get_install_dir() / 'modab_root_finder')
install_subdir('tests', install_dir: py.get_install_dir() / 'modab_root_finder')
