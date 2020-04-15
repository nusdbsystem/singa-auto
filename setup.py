from setuptools import setup, findall, find_packages


setup(name='singa-auto',
      version='0.1.9',
      description='The singa-auto',
      url='https://github.com/nusdbsystem/singa-auto.git',
      author='Naili',
      author_email='xingnaili14@gmail.com',
      license='Apache',
      packages=["singa_auto", "scripts"],
      include_package_data=True,
      install_requires=['docker',
                        'requests',
                        'numpy',
                        'pandas',
                        'temp',
                        'Pillow',
                        'requests-toolbelt',
                        'requests'
                        ],
      entry_points={
            'console_scripts': [
                  'admin=singa_auto:start_admin',
                  'predict=singa_auto:start_predictor',
                  'worker=singa_auto:start_worker',
            ],
      },
      zip_safe=False)
