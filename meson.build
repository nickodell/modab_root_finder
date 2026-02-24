project(
  'spin-modab-root-finder',
  'c', 'cython',
  version: '0.0.dev0',
  license: 'BSD-3',
  meson_version: '>= 0.64',
  default_options: [
    'buildtype=debugoptimized',
    'c_std=c99',
    'cpp_std=c++14',
  ],
)

cc = meson.get_compiler('c')

py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()

py.extension_module(
    'rect',
    'modab_root_finder/rect.pyx',
    dependencies: py_dep,
    override_options : ['cython_language=cpp'],
    install: true,
    include_directories: [],
    subdir: 'modab_root_finder',
)

subdir('modab_root_finder')
