/*
web has a .env specifying the custom settings for local dev:
===
PORT=$WEB_ADMIN_EXT_PORT
REACT_APP_API_POINT_HOST=$RAFIKI_ADDR
REACT_APP_API_POINT_PORT=$ADMIN_EXT_PORT
===
for local development, no need to source the env.sh or .env.
*/

// const adminHost = process.env.REACT_APP_API_POINT_HOST
// const adminPort = process.env.REACT_APP_API_POINT_PORT
const adminHost = 'http://ncrs.d2.comp.nus.edu.sg'
const adminPort = '3000'

const HTTPconfig = {
  // the client tells server data-type json is actually sent.
  HTTP_HEADER: {
    "Content-Type": "application/json",
  },
  UPLOAD_FILE: {
    "Content-Type": "multipart/form-data",
  },
  adminHost: `${adminHost}`,
  adminPort: `${adminPort}`,
  gateway: `http://${adminHost}:${adminPort}/`,
}

// start script's process.env.NODE_ENV = 'development';
// build script's process.env.NODE_ENV = 'production';
// if you run yarn start, the NODE_ENV will be development
const LocalGateways = {
  // NOTE: must append '/' at the end!
  local: "http://localhost:3000/",
  rafiki: "http://ncrs.d2.comp.nus.edu.sg:3000/",
  panda: "http://ncrs.d2.comp.nus.edu.sg:3000/",
}
if (process.env.NODE_ENV === "development") {
  // set the gateway for local development here:
  HTTPconfig.gateway = LocalGateways.panda
  // set the Host and Port for TrialDetails.js
  HTTPconfig.adminHost = `ncrs.d2.comp.nus.edu.sg`
  HTTPconfig.adminPort = `3000`
}
// otherwise, the docker build will set NODE_ENV to production

export default HTTPconfig
