import { Types } from "./actions"

const StatesToReset = {}

const initialState = {
  InferenceJobsList: [],
  ...StatesToReset,
}

export const InferenceJobsReducer = (state = initialState, action) => {
  switch (action.type) {
    case Types.POPULATE_INFERENCEJOB:
      return {
        ...state,
        InferenceJobsList: action.jobs.length === 0 ? [] : action.jobs,
      }
    default:
      return state
  }
}

export default InferenceJobsReducer
