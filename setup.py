from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='cnn-raccoon',
      version='0.5.0',
      description='CNN Raccoon: Analise your Convolulutional Neural Network with ease! (Tensorflow and PyTorch '
                  'supported)',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/lucko515/cnn-raccoon',
      author='Luka Anicin',
      author_email='luka.anicin@gmail.com',
      license='MIT',
      packages=find_packages(),
      classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Intended Audience :: Education",
            "Intended Audience :: Developers",
            "Intended Audience :: Information Technology",
            "Intended Audience :: Science/Research"
      ],
      python_requires='>=3',
      )
