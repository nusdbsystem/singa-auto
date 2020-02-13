import { Types } from "./actions"

const initialState = {
  // Available MODEL-List
  AvailableModelList: [],
}

export const ModelsReducer = (state = initialState, action) => {
  switch (action.type) {
    case Types.POPULATE_AVAILABLE_MODEL_LIST:
      return {
        ...state,
        AvailableModelList: action.AvailableModels.length === 0
          ? []
          : action.AvailableModels,
      }
    default:
      return state
  }
}

export default ModelsReducer
