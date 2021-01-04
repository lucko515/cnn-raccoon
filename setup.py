from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='cnn-raccoon',
      version='0.9.5',
      description='Create interactive dashboards for your Convolutional Neural Networks (CNNs) with a single line of code!',
      long_description=long_description,
      include_package_data=True,
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
