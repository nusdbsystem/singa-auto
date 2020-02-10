export const Types = {
  REQUEST_LS_DS: "Datasets/request_ls_ds",
  POPULATE_DS_LIST: "Datasets/populate_ds_list",
  CREATE_DATASET: "Datasets/create_dataset",
}

// LIST_DATASET{_ALL}
export const requestListDS = () => ({
  type: Types.REQUEST_LS_DS,
})

export const populateDSList = DatasetList => ({
  type: Types.POPULATE_DS_LIST,
  DatasetList,
})

// moved the axios and uploadprogress and formstate
// to web/src/containers/Datasets/UploadDataset.js Jan03-2020
// export const postCreateDataset = (name, task, file, dataset_url) => ({
//   type: Types.CREATE_DATASET,
//   name,
//   task,
//   file,
//   dataset_url,
// })
