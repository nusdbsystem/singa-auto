import { takeLatest, call, put, fork, select } from "redux-saga/effects"
import { showLoading, hideLoading } from "react-redux-loading-bar"
import * as actions from "../containers/InferenceJobs/actions"
import { notificationShow } from "../containers/Root/actions.js"
import * as api from "../services/ClientAPI"
import { getToken, getUserId } from "./utils"

// Watch action request InferenceJobs list and run generator getInferenceJobsList
function* watchGetInferenceJobsListRequest() {
  yield takeLatest(actions.Types.FETCH_GET_INFERENCEJOB, getInferenceJobsList)
}

/* for List InferenceJobs command */
function* getInferenceJobsList() {
  try {
    console.log("Start to load InferenceJobs")
    yield put(showLoading())
    const token = yield select(getToken)
    const user_id = yield select(getUserId)
    const InferenceJobs = yield call(api.getInferenceJob, { user_id }, token)
    console.log("InferenceJobs loaded", InferenceJobs.data)
    yield put(actions.populateInferenceJob(InferenceJobs.data))
    yield put(hideLoading())
  } catch (e) {
    console.error(e.response)
    console.error(e)
    alert(e.response.data)
    yield put(notificationShow("Failed to Fetch Inference Jobs List"))
    // TODO: implement notification for success and error of api actions
    // yield put(actions.getErrorStatus("failed to deleteUser"))
  }
}

// watch action call POST CREATE INFERENCEJOB => call api.create inference job
function* watchCreateInferenceJobRequest() {
  yield takeLatest(actions.Types.POST_CREATE_INFERENCEJOB, createInferenceJob)
}

function* createInferenceJob(action) {
  const { app, appVersion, budget } = action
  try {
    const token = yield select(getToken)
    yield call(api.createInferenceJob, app, appVersion, budget, token)
    yield put(notificationShow("Create Inference Job Successfully")) // no need to write test for this
  } catch (e) {
    console.error(e.response)
    console.error(e)
    alert(e.response.data)
    yield put(notificationShow("Failed to Create Inference Job"))
  }
}

// fork is for process creation, run in separate processes
export default [
  fork(watchGetInferenceJobsListRequest),
  fork(watchCreateInferenceJobRequest),
]
