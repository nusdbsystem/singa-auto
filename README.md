<!--
    Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.
-->

# SINGA-Auto

*SINGA-Auto* is a distributed system that trains machine learning (ML) models and deploys trained models, built with ease-of-use in mind. To do so, it leverages on automated machine learning (AutoML).

Read SINGA-Auto's full documentation at https://singa-auto.readthedocs.io/en/latest

## Quick Setup

Prerequisites: MacOS or Linux environment

1. Install Docker 18 ([Ubuntu](https://docs.docker.com/install/linux/docker-ce/ubuntu/), [MacOS](https://docs.docker.com/docker-for-mac/install/)) and, if required, add your user to `docker` group ([Linux](https://docs.docker.com/install/linux/linux-postinstall/>))

2. Install Python 3.6 ([Ubuntu](http://ubuntuhandbook.org/index.php/2017/07/install-python-3-6-1-in-ubuntu-16-04-lts/), [MacOS](https://www.python.org/downloads/mac-osx/))

3. Clone this project (e.g. with [Git](https://git-scm.com/downloads>))

4. Setup  SINGA-Auto's complete stack with the setup script:

   if use docker swarm mode, use this script: 

    ```sh
    bash scripts/docker_swarm/start.sh
    ```
    
   if use kubernetes mode, use this script:
   
    ```sh
    bash scripts/kubernetes/start.sh
    ```

  To completely destroy  SINGA-Auto's stack:

   if use docker swarm mode, use this script:
   
    ```sh
    bash scripts/docker_swarm/stop.sh
    ```
    
   if user kubernetes mode, use this script:
   
    ```sh
    bash scripts/kubernetes/stop.sh
    ```

More instructions are available in [SINGA-Auto's Developer Guide](https://singa-auto.readthedocs.io/en/latest).


## Issues

Report any issues at [SINGA-Auto's Github Issues](https://github.com/nusdbsystem/singa-auto/issues)

## Jira

View our development records at [SINGA-Auto's Jira Issues](https://team-1628053718937.atlassian.net/jira/software/projects/SINGAAUTO/issues/)

## Acknowledgements

The research is supported by the National Research Foundation, Prime Minister’s Office, Singapore under its National Cybersecurity R\&D Programme (Grant No. NRF2016NCR-NCR002-020), National Natural Science Foundation of China (No. 61832001), National Key Research and Development Program of China  (No. 2017YFB1201001), China Thousand Talents Program for Young Professionals (3070011 181811).
