import sys
from setuptools import Extension
from setuptools import setup
from setuptools import find_packages
from setuptools import dist
import os
dist.Distribution().fetch_build_eggs(['numpy>=1.10'])
import numpy as np

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths
  
extra_files = package_files('nltk/sejong')
extra_files = extra_files+package_files('nltk/tag')

module1 = Extension("nltk.tag.espresso.libs.network", 
                               ["nltk/tag/espresso/libs/network.c"],
                               include_dirs=['.', np.get_include()])

setup(
  name='nltk',
  version='1.1.4',
	url='https://github.com/cwnu-air/NLTKo.git',
  packages=find_packages(exclude=[]),
  install_requires=['regex==2020.7.14','numpy','requests','beautifulSoup4'],
  package_data={'': extra_files},
	ext_modules=[module1],
  include_package_data=True
)
