import { takeLatest, call, put, fork, select } from "redux-saga/effects"
import { showLoading, hideLoading, resetLoading } from "react-redux-loading-bar"
import * as actions from "../containers/Datasets/actions"
import * as ConsoleActions from "../containers/ConsoleAppFrame/actions"
import { notificationShow } from "../containers/Root/actions.js"
import * as api from "../services/ClientAPI"
import { getToken } from "./utils"

// List Datasets
function* watchGetDSListRequest() {
  yield takeLatest(actions.Types.REQUEST_LS_DS, getDatasetList)
}

/* for List Dataset command */
function* getDatasetList() {
  try {
    yield put(showLoading())
    const token = yield select(getToken)
    const DSList = yield call(api.requestDatasetList, {}, token)
    console.log("DSsagas | getDatasetList(): ", DSList)
    yield put(actions.populateDSList(DSList.data))
    yield put(hideLoading())
  } catch (e) {
    console.error(e.response)
    console.error(e)
    yield put(notificationShow("Failed to Fetch DatasetList"))
    // TODO: implement notification for success and error of api actions
    // yield put(actions.getErrorStatus("failed to deleteUser"))
  }
}

// function* watchPostDatasetsRequest() {
//   yield takeLatest(actions.Types.CREATE_DATASET, createDataset)
// }

// // moved the axios and uploadprogress and formstate
// // to web/src/containers/Datasets/UploadDataset.js Jan03-2020
// function* createDataset(action) {
//   const { name, task, file, dataset_url } = action
//   try {
//     yield put(showLoading())
//     const token = yield select(getToken)
//     const postDSres = yield call(api.postCreateDataset, name, task, file, dataset_url, token)
//     console.log("uploadDS res: ", postDSres)
//     console.log("Create Dataset success")
//     yield alert("Create Dataset success")
//     yield put(notificationShow("Create Dataset Success")) // no need to write test for this
//     yield push("console/datasets/list-datasets")
//     yield put(hideLoading())
//   } catch (e) {
//     console.error(e.response)
//     console.error(e)
//     console.error(e.response.data)
//     yield put(notificationShow("Failed to Create Dataset"))
//   }
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
  fork(watchGetDSListRequest),
  //fork(watchPostDatasetsRequest),
  fork(watchResetLoadingBar),
]
