import { Types } from "./actions"

const initialState = {
  headerTitle: "Overview",
}

export const ConsoleAppFrame = (state = initialState, action) => {
  switch (action.type) {
    case Types.CHANGE_HEADER_TITLE:
      return {
        ...state,
        headerTitle: action.headerTitle,
      }
    default:
      return state
  }
}
