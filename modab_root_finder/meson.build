py.extension_module(
  '_core',
  'coremodule.c',
  install: true,
  subdir: 'modab_root_finder'
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
