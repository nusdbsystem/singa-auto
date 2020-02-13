import { takeLatest, call, put, fork, select } from "redux-saga/effects"
import { showLoading, hideLoading, resetLoading } from "react-redux-loading-bar"
import * as actions from "../containers/Models/actions"
import * as ConsoleActions from "../containers/ConsoleAppFrame/actions"
import { notificationShow } from "../containers/Root/actions.js"
import * as api from "../services/ClientAPI"
import { getToken } from "./utils"

// Watch action request {available} Model list and run generator getModelList
function* watchGetAvailableModelListRequest() {
  yield takeLatest(
    actions.Types.REQUEST_AVAILABLE_MODEL_LIST,
    getAvailableModelList)
}

/* for List Available Model command */
function* getAvailableModelList() {
  try {
    // console.log("Start to load available models")
    yield put(showLoading())
    const token = yield select(getToken)
    // get models/available need auth+task
    // for NOW, task can be left blank, since we
    // only using available?task=IMAGE_CLASSIFICATION
    const models = yield call(api.getAvailableModels, {}, token)
    console.log("Available Model loaded", models.data)
    yield put(actions.populateAvailableModelList(models.data))
    yield put(hideLoading())
  } catch (e) {
    console.error(e.response)
    console.error(e)
    yield put(notificationShow("Failed to Fetch Available Model List"))
    // TODO: implement notification for success and error of api actions
    // yield put(actions.getErrorStatus("failed to deleteUser"))
  }
}

// TO BE IMPLEMENTED

// function* watchPostModelsRequest() {
//     yield takeLatest(actions.Types.CREATE_Model, createModel)
// }

// function* createModel(action) {
//     const {name, task, file, Model_url} = action
//     try {
//         const token = yield select(getToken)
//         yield call(api.postCreateModel, name, task, file, Model_url, token)
//         console.log("Create Model success")
//         yield put(notificationShow("Create ModelList Successfully")); // no need to write test for this
//     } catch(e) {
//         console.error(e.response)
//         console.error(e)
//         console.error(e.response.data)
//         yield put(notificationShow("Failed to Create Dataset"));
//     }
// }

/* reset loadingBar caused by List Dataset command */
function* callResetLoadingBar() {
  try{
    yield put(resetLoading())
  } catch(e) {
    console.error(e)
  }
}

function* watchResetLoadingBar() {
  yield takeLatest(ConsoleActions.Types.RESET_LOADING_BAR, callResetLoadingBar)
}

// fork is for process creation, run in separate processes
export default [
  fork(watchGetAvailableModelListRequest),
  fork(watchResetLoadingBar),
]
