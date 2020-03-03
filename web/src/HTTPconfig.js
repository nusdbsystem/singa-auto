// start script's process.env.NODE_ENV = 'development';
// build script's process.env.NODE_ENV = 'production';
// default as development

/*
web has a .env specifying the custom settings for local dev:
PORT=$WEB_ADMIN_EXT_PORT
REACT_APP_API_POINT_HOST=$RAFIKI_ADDR
REACT_APP_API_POINT_PORT=$ADMIN_EXT_PORT
===
for local development, no need to source the env.sh or .env.
*/

const adminHost = process.env.REACT_APP_API_POINT_HOST
const adminPort = process.env.REACT_APP_API_POINT_PORT

const HTTPconfig = {
  // the client tells server data-type json is actually sent.
  HTTP_HEADER: {
    "Content-Type": "application/json",
  },
  UPLOAD_FILE: {
    "Content-Type": "multipart/form-data",
  },
  // need a working server for axios uploadprogress to work
  // gateway: "http://localhost:5000/",
  // gateway: "http://ncrs.d2.comp.nus.edu.sg:3000/"
  adminHost: `${adminHost}`,
  adminPort: `${adminPort}`,
  gateway: `http://${adminHost}:${adminPort}/`,
}

if (process.env.NODE_ENV === "development") {
  // localhost:3000 is the port exposed by
  // docker rafiki admin
  HTTPconfig.gateway = "http://localhost:3000/"
}

export default HTTPconfig
