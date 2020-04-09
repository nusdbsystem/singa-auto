export const Types = {
  // ==== MODELS =====
  // ASYNC
  REQUEST_AVAILABLE_MODEL_LIST: "Models/request_available_model_list",
  POST_CREATE_MODEL: "Models/post_create_model",
  // SYNC
  POPULATE_AVAILABLE_MODEL_LIST: "Models/populate_available_model_list",
}

// List Available Models
export function requestAvailableModelList() {
  return {
    type: Types.REQUEST_AVAILABLE_MODEL_LIST,
  }
}

export function populateAvailableModelList(AvailableModels) {
  return {
    type: Types.POPULATE_AVAILABLE_MODEL_LIST,
    AvailableModels,
  }
}

// axios and upload in UploadModel.js
