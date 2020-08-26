#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#

from setuptools import setup

setup(name='singa-auto',
      version='0.3.4',
      description='The SINGA-Auto',
      url='https://github.com/nusdbsystem/singa-auto.git',
      author='Naili',
      author_email='xingnaili14@gmail.com',
      license='Apache',
      packages=["singa_auto", "scripts"],
      include_package_data=True,
      install_requires=[
          'docker', 'requests', 'numpy', 'pandas', 'temp', 'Pillow',
          'requests-toolbelt', 'requests'
      ],
      entry_points={
          'console_scripts': [
              'sago=singa_auto:start_all',
              'sastop=singa_auto:stop_all',
              'saclean=singa_auto:clean',
              'admin=singa_auto:start_admin',
              'predict=singa_auto:start_predictor',
              'worker=singa_auto:start_worker',
          ],
      },
      zip_safe=False)
