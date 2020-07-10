# SINGA-auto Dashboard

> Ensure you are in the `REPO_ROOT/web/` directory

## Set up local development

### Set up Docker orchestration for backend
- install docker [setupDocker.md](./docsWebDev/seupDocker.md)
- run `REPO_ROOT/scripts/start.sh`, this will start all the docker-swarm Kafka, Redis, Postgres, Flask, and web
- **for local web development, we will only make use of the non-web docker images**
- change the API and port in the `./src/HTTPconfig.js`:
```js
if (process.env.NODE_ENV === "production") {
  //HTTPconfig.gateway = "http://13.229.126.135/"
  HTTPconfig.gateway = `http://${adminHost}:${adminPort}/`
}
```

### Install Node Packages

```sh
yarn install
```

### Run the create-react-app local development
```
yarn start
```
Then open `http://localhost:65432/` to see your app in a browser. If you want to use another port, specify it before the `yarn start`:
```sh
PORT=<your port number> yarn start
```

In this react-app, you can run several commands:
```sh
yarn start
# Starts the development server.

yarn build
# Bundles the app into static files for production.

yarn test
# Starts the test runner.

yarn eject
# Removes this tool and copies build dependencies, configuration files and scripts into the app directory. If you do this, you can’t go back!
# do not eject the app
```

### Push local changes
- update change to be reflect in the docker image:
```sh
singa-auto$ bash scripts/build_images.sh
#(this will create a new ubuntu-based docker image)
```
- list docker containers
```sh
singa-auto$ docker container ls
```

===

## How to setup Api End Point (environment parameters)

```sh
singa-auto/web$ source ../.env.sh
```

```
singa-auto/web $ vim .env
```

```
PORT=$WEB_ADMIN_EXT_PORT
NODE_PATH=./src
REACT_APP_API_POINT_HOST=$RAFIKI_ADDR
REACT_APP_API_POINT_PORT=$ADMIN_EXT_PORT
```

```
export DOCKER_SWARM_ADVERTISE_ADDR=10.0.0.125
export RAFIKI_VERSION=0.3.0
export RAFIKI_ADDR=ncrs.d2.comp.nus.edu.sg
```

## How to SET input path

- https://facebook.github.io/create-react-app/docs/importing-a-component#absolute-imports

## How to SET docker to auto recompile when file changed
- https://stackoverflow.com/questions/46379727/react-webpack-not-rebuilding-when-edited-file-outside-docker-container-on-mac

```
ENV CHOKIDAR_USEPOLLING=true
ENV CHOKIDAR_INTERVAL=1000
```

## How to TEST the app

```
yarn test
```

https://github.com/facebook/jest/issues/3254

If you cannot test on Ubuntu, maybe it is because jest is watching too many files.

```
echo fs.inotify.max_user_watches=524288 | sudo tee -a /etc/sysctl.conf && sudo sysctl -p
```

## How the environment variable is processed in React

- https://facebook.github.io/create-react-app/docs/adding-custom-environment-variables

### migrate from `react-testing-library` to `@testing-library/react`

```
react-testing-library has moved to @testing-library/react. Please uninstall react-testing-library and install @testing-library/react instead, or use an older version of react-testing-library. Learn more about this change here: https://github.com/testing-library/dom-testing-library/issues/260 Thanks! :)
```

### Test library

## API Endpoints

## users

### post `/token`

header: content_type = 'application/json'

#### request
``` json
{
	"email": "superadmin@singaauto",
	"password": "singa_auto"
}
```

#### response

``` json
{
    "token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoiYWUxN2FhOWYtNWRmNi00MWVjLTgzNDEtNWM2N2NjYmU0NTAxIiwidXNlcl90eXBlIjoiU1VQRVJBRE1JTiJ9.dxWzHxJ2nz_cX0oeiyOQY3wWD4irlpT47e-usfrVSWw",
    "user_id": "ae17aa9f-5df6-41ec-8341-5c67ccbe4501",
    "user_type": "SUPERADMIN"
}
```


## Datasets

### get `/datasets`

```

```

### psot `datasets`

```

```

## models

### post `models` (working)

should use muitlform but does not need to set it because the browser would set it automatically.

### get `models`

#### header 

```
    'Authorization': `Bearer ${this._token}`
```

#### response

``` json
[
    {
        "access_right": "PRIVATE",
        "datetime_created": "Wed, 29 May 2019 15:05:56 GMT",
        "dependencies": {
            "tensorflow": "1.12.0"
        },
        "docker_image": "rafikiai/rafiki_worker:0.0.9",
        "model_class": "TfFeedForward",
        "name": "TfFeedForward_GOZ9JANFQVVQT6LX",
        "task": "IMAGE_CLASSIFICATION",
        "user_id": "ae17aa9f-5df6-41ec-8341-5c67ccbe4501"
    },
    {
        "access_right": "PRIVATE",
        "datetime_created": "Wed, 29 May 2019 15:05:56 GMT",
        "dependencies": {
            "scikit-learn": "0.20.0"
        },
        "docker_image": "rafikiai/rafiki_worker:0.0.9",
        "model_class": "SkDt",
        "name": "SkDt_GOZ9JANFQVVQT6LX",
        "task": "IMAGE_CLASSIFICATION",
        "user_id": "ae17aa9f-5df6-41ec-8341-5c67ccbe4501"
    },
    {
        "access_right": "PRIVATE",
        "datetime_created": "Wed, 29 May 2019 15:10:27 GMT",
        "dependencies": {
            "tensorflow": "1.12.0"
        },
        "docker_image": "rafikiai/rafiki_worker:0.0.9",
        "model_class": "TfFeedForward",
        "name": "TfFeedForward_RY0DI2PV26UMPJ2L",
        "task": "IMAGE_CLASSIFICATION",
        "user_id": "ae17aa9f-5df6-41ec-8341-5c67ccbe4501"
    },
    {
        "access_right": "PRIVATE",
        "datetime_created": "Wed, 29 May 2019 15:10:27 GMT",
        "dependencies": {
            "scikit-learn": "0.20.0"
        },
        "docker_image": "rafikiai/rafiki_worker:0.0.9",
        "model_class": "SkDt",
        "name": "SkDt_RY0DI2PV26UMPJ2L",
        "task": "IMAGE_CLASSIFICATION",
        "user_id": "ae17aa9f-5df6-41ec-8341-5c67ccbe4501"
    },
    {
        "access_right": "PRIVATE",
        "datetime_created": "Thu, 30 May 2019 06:07:09 GMT",
        "dependencies": {
            "tensorflow": "1.12.0"
        },
        "docker_image": "rafikiai/rafiki_worker:0.0.9",
        "model_class": "TfFeedForward",
        "name": "TfFeedForward",
        "task": "IMAGE_CLASSIFICATION",
        "user_id": "ae17aa9f-5df6-41ec-8341-5c67ccbe4501"
    }
]
```

## trainJobs

## post `/train_job`

### requrest

#### header 
```
     'Authorization': `Bearer ${this._token}`
     'Content-type': 'application/json'
```
#### body

```
{
    "app":"fashion_mnist_app",
    "task": "IMAGE_CLASSIFICATION",
    "train_dataset_uri":"https://github.com/nginyc/rafiki-datasets/blob/master/fashion_mnist/fashion_mnist_for_image_classification_train.zip?raw=true",
    "test_dataset_uri":"https://github.com/nginyc/rafiki-datasets/blob/master/fashion_mnist/fashion_mnist_for_image_classification_test.zip?raw=true",
        "budget": { "MODEL_TRIAL_COUNT": 2 }
    "models" ["TfFeedForward"]
}
```

### response

```
{
    "app": "fashion_mnist_app",
    "app_version": 1,
    "id": "df266b4c-5a04-43f8-a751-230b5c7fec8f"
}
```

### get `/train_jobs?user_id={user_id}`

#### header

```
    'Authorization': `Bearer ${this._token}`
```

#### response

```
[
    {
        "app": "image_classification_app_5CWOI56WR7ABMKJV",
        "app_version": 1,
        "budget": {
            "ENABLE_GPU": 0,
            "MODEL_TRIAL_COUNT": 2
        },
        "datetime_started": "Tue, 26 Mar 2019 04:29:30 GMT",
        "datetime_stopped": "Tue, 26 Mar 2019 04:30:40 GMT",
        "id": "db738d0e-9a5f-48af-bde1-28b1c71bee02",
        "status": "STOPPED",
        "task": "IMAGE_CLASSIFICATION",
        "test_dataset_uri": "https://github.com/nginyc/rafiki-datasets/blob/master/fashion_mnist/fashion_mnist_for_image_classification_test.zip?raw=true",
        "train_dataset_uri": "https://github.com/nginyc/rafiki-datasets/blob/master/fashion_mnist/fashion_mnist_for_image_classification_train.zip?raw=true"
    },
	......
]
```

### post `/inference_jobs`

#### request 

```
{ 
  'user_id': user_id
}
```

#### response

```
{

}
```

### get `/inference_jobs` 

```
{ 
  'user_id': user_id
}
```

####  response 

```
{
app:
"fashion_minist_app"
app_version:
2
datetime_started:
"Tue, 13 Aug 2019 12:37:10 GMT"
datetime_stopped:
"Tue, 13 Aug 2019 12:52:08 GMT"
id:
"85075d4a-eeda-42de-bb9a-5bc007637838"
predictor_host:
"ncrs.d2.comp.nus.edu.sg:53589"
status:
"STOPPED"
train_job_id:
"123b32aa-9c3e-449c-b2da-031d3123b631"
}
```

#### NOTES:

Not considering moving this project to Typescript, because compile time is quite long.

https://stackoverflow.com/questions/47508564/migrating-create-react-app-from-javascript-to-typescript

#### Not rendering the page when url is changed

https://github.com/supasate/connected-react-router/issues/230

#### TODO

- V0.3.0
    - User able to stop inference job on web
    - 

- Add test to the whole app
    - Unit test
        - Add test to sagas
        - Add test to redux action and reducer
    - Integreted Tests
    - End to end test
- Implement Authorization [Done]
- Redesign the Datasets Page with Gridsystem [Done]
- Redirect to mainpage
- Pagination
- Refractor ClientAPI, AuthAPI to Client object (Class) with user_id and token



# For Developers

This app use redux to manage the state and the state shape is like this 

```
{
    "datasets": [{
        // Datasets details accodring to the REST API
    }]
    "trials": [{
        // Trial details according to the REST API
    }]
    "appUtils": [{

    }]
}

```
