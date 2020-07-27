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

## Backend API Endpoints

When we develop the web UI, we usually use [Axios](https://github.com/axios/axios) to send request to the backend API endpoint, or use [Curl](https://github.com/curl/curl) to test.

### Tips for using Axois

- https://kapeli.com/cheat_sheets/Axios.docset/Contents/Resources/Documents/index
- https://simplecheatsheet.com/tag/axios-cheat-sheet 

### Tips for using Curl

- https://gist.github.com/subfuzion/08c5d85437d5d4f00e58
- https://gist.github.com/joyrexus/85bf6b02979d8a7b0308

### SINGA-Auto Backend Restful API

- https://github.com/nusdbsystem/singa-auto/wiki/SinaAuto-Api-Documents

### SINGA-Auto Predictor Restful API

- https://github.com/nusdbsystem/singa-auto/wiki/SingaAuto-Predictor-API
