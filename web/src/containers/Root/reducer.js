import { Types } from "./actions"


const initialState = {
    token: null,
    user_id: null, 
    error: null, 
    loading: false,
    notification: {
      show: false,
      message: ""
    },
}


export const Root = (state = initialState, action) => {
  switch (action.type) {
    // login menu on appbar (removed from reducer)
    // for authentications
    case Types.AUTH_START:
      return {
        ...state,
        error: null,
        loading: true
      }
    case Types.AUTH_SUCCESS:
      return {
        ...state,
        token: action.token,
        user_id: action.user_id,
        error: null,
        loading: false
      }
    case Types.AUTH_FAIL:
      return {
        ...state,
        error: action.error,
        loading: false
      }
    case Types.AUTH_LOGOUT:
      return {
        ...state,
        token: null
      }
    // for notification area
    case Types.NOTIFICATION_SHOW:
      return {
        ...state,
        notification: {
          show: true,
          message: action.message
        }
      };
    case Types.NOTIFICATION_HIDE:
      return {
        ...state,
        notification: {
          show: false,
          message: ""
        }
      };
    default:
      return state;
  }
};
