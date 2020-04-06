export const Types = {
  FETCH_GET_INFERENCEJOB: "InferenceJob/fetch_get_inferencejob",
  POST_CREATE_INFERENCEJOB: "InferenceJob/post_create_inferencejob",

  GET_RUNNING_INFERENCEJOB: "InferenceJob/get_running_inferencejob",
  SELECT_INFERENCEJOB: "InferenceJob/select_inferencejob",

  POPULATE_INFERENCEJOB: "InferenceJob/display_infenrencejob",
}

export const fetchGetInferencejob = () => {
  return {
    type: Types.FETCH_GET_INFERENCEJOB,
  }
}

export const postCreateInferenceJob = (app, appVersion, budget) => {
  return {
    type: Types.POST_CREATE_INFERENCEJOB,
    app,
    appVersion,
    budget,
  }
}

// sync
export const populateInferenceJob = jobs => {
  return {
    type: Types.POPULATE_INFERENCEJOB,
    jobs,
  }
}
