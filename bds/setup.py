from setuptools import find_packages, setup

setup(name='bds',
      version='0.1.0',
      author='Scangos Interventional Psychiatry Lab',
      author_email='pmdaly12@gmail.com',
      packages=find_packages(exclude=('test', 'docs')),
      include_package_data=True,
      license='MIT',
      long_description=open('README.md').read(),
      install_requires=[
          'numpy',
          'pandas',
          'supereeg'
          ]
      )
