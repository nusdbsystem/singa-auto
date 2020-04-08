from setuptools import setup

setup(name='singa_auto',
      version='0.1',
      description='The singa_auto',
      url='https://github.com/nusdbsystem/singa-auto.git',
      author='Naili',
      author_email='xingnaili14@gmail.com',
      license='Apache',
      packages=['singa_auto'],
      install_requires=['docker',
                        'requests',
                        'numpy',
                        'pandas',
                        'temp',
                        'Pillow',
                        'requests-toolbelt'
                        ],
      scripts=['scripts/sa',
               ],
      zip_safe=False)
