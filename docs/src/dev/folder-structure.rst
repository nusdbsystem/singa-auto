.. _`folder-structure`:

Folder Structure
====================================================================

- `singa_auto/`

    SINGA-Auto's Python package 

    - `admin/`

        SINGA-Auto's static Admin component

    - `advisor/`

        SINGA-Auto's advisors

    - `client/`

        SINGA-Auto's client-side SDK

        .. seealso:: :class:`singa_auto.client`

    - `worker/`

        SINGA-Auto's train, inference & advisor workers
    
    - `predictor/`

        SINGA-Auto's predictor

    - `meta_store/`

        Abstract data access layer for singa_auto's main metadata store (backed by PostgreSQL)
    
    - `param_store/`

        Abstract data access layer for SINGA-Auto's store of model parameters (backed by filesystem)

    - `data_store/`

        Abstract data access layer for SINGA-Auto's store of datasets (backed by filesystem)

    - `cache/`

        Abstract data access layer for SINGA-Auto's temporary store of model parameters, train job metadata and queries & predictions in train & inference jobs (backed by Redis)

    - `container/`

        Abstract access layer for dynamic deployment of workers 

    - `utils/`

        Collection of SINGA-Auto-internal utility methods (e.g. for logging, authentication)

    - `model/`

        Definition of abstract :class:`singa_auto.model.BaseModel` that all SINGA-Auto models should extend, programming 
        abstractions used in model development, as well as a collection of utility methods for model developers 
        in the implementation of their own models
    
    - `constants.py`

        SINGA-Auto's programming abstractions & constants (e.g. valid values for user types, job statuses)

- `web/`

    SINGA-Auto's Web Admin component
    
- `dockerfiles/`
    
    Stores Dockerfiles for customized components of SINGA-Auto 

- `examples/`
    
    Sample usage code for SINGA-Auto, such as standard models, datasets dowloading and processing codes, sample image/question data, and quick test code

- `docs/`

    Source documentation for SINGA-Auto (e.g. Sphinx documentation files)

- `test/`

    Test code for SINGA-Auto

- `scripts/`

    Shell & python scripts for initializing, starting and stopping various components of SINGA-Auto's stack

    - `docker_swarm/`

        Containing server environment settings and scripts for running Docker

    - `kubernetes/`

        Containing server environment settings and scripts for running Kubernetes

    - `.base_env.sh`

        Stores configuration variables for SINGA-Auto

- `log_minitor/`

    Dockerfile and configurations for elasticsearch and logstash

- `singa_auto_scheduler/`

    Dockerfiles and configurations for scheduler and monitor
