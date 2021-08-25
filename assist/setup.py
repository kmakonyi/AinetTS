from setuptools import setup, find_packages

setup(name='assist',
      version='0.1',
      package_dir={"": "src"},
      packages=find_packages('src'),
      install_requires=['pandas>1.3, appdirs'],
      test_requires=['pytest']
      )
