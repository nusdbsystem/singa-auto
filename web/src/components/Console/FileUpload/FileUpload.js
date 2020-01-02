import React, { Fragment, useState } from 'react';
//import Message from './Message';
import UploadProgressBar from './UploadProgressBar';
import axios from 'axios';
import HTTPconfig from "HTTPconfig"

import FileDropzone from "./FileDropzone"

const FileUpload = () => {
  // populate the files state from FileDropzone
  const [selectedFiles, setSelectedFiles] = useState([])

  const [message, setMessage] = useState('')
  const [uploadPercentage, setUploadPercentage] = useState(0);

  const [uploadSuccess, setUploadStatus] = useState(false)

  // for dataset
  const [name, setName] = useState('Sample-DS-1')
  const [task, setTask] = useState("IMAGE_CLASSIFICATION")

  const onDrop = files => {
    // file input, can access the file props
    // files is an array
    // files[0] is the 1st file we added
    // console.log(event.target.files[0])
    console.log("onDrop called, acceptedFiles: ", files)
    setSelectedFiles(files)
  }

  const handleRemoveCSV = () => {
    setSelectedFiles([])
    console.log("file removed")
  }

  const onSubmit = async e => {
    e.preventDefault();
    // construct form data for sending
    // FormData() is default native JS object
    const formData = new FormData()
    // append(<whatever name>, value, <namePropterty>)
    console.log("selectedFiles[0]: ", selectedFiles[0])
    // flask createDS endpoint will look for
    // 'dataset' in request.files
    formData.append("dataset", selectedFiles[0])
    formData.append("name", name)
    formData.append("task", task)

    try {
      const res = await axios.post(
        `${HTTPconfig.gateway}api/upload-csv`,
        formData, 
        {
          headers: {
            'Content-Type': 'multipart/form-data'
          },
          onUploadProgress: progressEvent => {
            // progressEvent will contain loaded and total
            let percentCompleted = parseInt(
              Math.round( (progressEvent.loaded * 100) / progressEvent.total )
            )
            console.log("From EventEmiiter, file Uploaded: ", percentCompleted)
            setUploadPercentage(percentCompleted);
          }
        }
      );
      // res.data is the object sent back from the server
      console.log("file uploaded, res.data: ", res.data)

      setUploadStatus(true)

      setMessage(selectedFiles[0]["name"] + ' Uploaded')
    } catch (err) {
      console.error(err, "error")
      setUploadStatus(false)
      setMessage("upload failed")
    }
  };

  return (
    <Fragment>
      <FileDropzone
        files={selectedFiles}
        onCsvDrop={onDrop}
        onRemoveCSV={handleRemoveCSV}
      />
      <form onSubmit={onSubmit}>
        <input
          type='submit'
          value='Upload'
        />
      </form>
      <UploadProgressBar
        percentCompleted={uploadPercentage}
        fileName={
          selectedFiles.length !== 0
          ? selectedFiles[0]["name"]
          : ""
        }
        uploaded={uploadSuccess}
      />
      {message ? <span>{message}</span> : null}
    </Fragment>
  );
};

export default FileUpload;