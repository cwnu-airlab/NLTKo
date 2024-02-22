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
  version='1.2.9',
	url='https://modi.changwon.ac.kr/air_cwnu/nlp_tool/nltk_ko.git',
  packages=find_packages(exclude=[]),
  python_requires='>=3.7',
  install_requires=[
    'regex==2020.7.14',
    "safetensors",
    "torch",
    "transformers>=4.8.2",
    "datasets",
    "faiss-cpu>=1.7.3",
    "bert_score",
    "fasttext",
    "pandas",
    "joblib",

    
    'numpy==1.23',
    'requests',
    'beautifulSoup4',
    'sentencepiece',
    'protobuf',
    

    'scikit-learn>=0.22.1',
    'tqdm>=4.40.0',

    ],
  package_data={'': extra_files},
	ext_modules=[module1],
  include_package_data=True,

  tests_require=["pytest"],
  classifiers=[
      "Programming Language :: Python :: 3.7",
      "Programming Language :: Python :: 3.8",
      "Programming Language :: Python :: 3.9",
      "Programming Language :: Python :: 3.10",
      "License :: OSI Approved :: MIT License",
      "Operating System :: OS Independent",
      "Typing :: Typed",
  ],
  keywords=[
      "string matching",
      "pattern matching",
      "edit distance",
      "string to string correction",
      "string to string matching",
      "Levenshtein edit distance",
      "Hamming distance",
      "Damerau-Levenshtein distance",
      "Jaro-Winkler distance",
      "longest common subsequence",
      "longest common substring",
      "dynamic programming",
      "approximate string matching",
      "semantic similarity",
      "natural language processing",
      "NLP",
      "information retrieval",
      "rouge",
      "sacrebleu",
      "bertscore",
      "bartscore",
      "fasttext",
      "glove",
      "cosine similarity",
      "Smith-Waterman",
      "Needleman-Wunsch",
      "Hirschberg",
      "Karp-Rabin",
      "Knuth-Morris-Pratt",
      "Boyer-Moore",
  ],
)
