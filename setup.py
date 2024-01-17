from setuptools import setup, find_packages

requires = ['matplotlib >=3.8.0', 'numpy', 'pandas', 'scikit-learn', 'scikit-image', 'scipy',
            'xarray', 'cfgrib', 'h5netcdf', 'rioxarray', 'pyresample', 'folium', 'contextily',
            'enpt', 'satpy', 'algotom', 'dem_stitcher', 'aiohttp', 'spacetrack', 'roaring-landmask'
            ]
extras_require = {
    'doc': ['sphinx', 'sphinx_rtd_theme'],
}
all_extras = []
for extra_deps in extras_require.values():
    all_extras.extend(extra_deps)
extras_require["all"] = list(set(all_extras))

setup(
    name='hypergas',
    description= 'Python package for hyperspectral satellite imaging of trace gases',
    version= '0.1.0',
    author = 'Xin Zhang',
    packages=find_packages(),
    install_requires=requires,
    extras_require=extras_require,
    python_requires='>=3.10',
    )
