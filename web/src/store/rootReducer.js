import { combineReducers } from "redux"

// LoadingBar
import { loadingBarReducer } from "react-redux-loading-bar"

import { Root } from "../containers/Root/reducer"
import { ConsoleAppFrame } from "../containers/ConsoleAppFrame/reducer"

// Models
import { DatasetsReducer } from "../containers/Datasets/reducer"
import { JobsReducer } from "containers/Jobs/reducer"
import { ModelsReducer } from "containers/Models/reducer"
import { InferenceJobsReducer } from "containers/InferenceJobs/reducer"

// Router
import { connectRouter } from "connected-react-router"

const rootReducer = history => {
  return combineReducers({
    loadingBar: loadingBarReducer,
    router: connectRouter(history),
    // app reducers:
    Root,
    ConsoleAppFrame,
    DatasetsReducer,
    JobsReducer,
    ModelsReducer,
    InferenceJobsReducer,
  })
}

export default rootReducer
