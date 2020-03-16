import React from "react"
import PropTypes from "prop-types"

import axios from 'axios';

import { connect } from "react-redux"
import { compose } from "redux"
import { push } from "connected-react-router"

import * as ConsoleActions from "../ConsoleAppFrame/actions"
import * as actions from "./actions"

import { withStyles } from "@material-ui/core/styles"
import Table from '@material-ui/core/Table';
import TableBody from '@material-ui/core/TableBody';
import TableCell from '@material-ui/core/TableCell';
import TableRow from '@material-ui/core/TableRow';
import TableHead from '@material-ui/core/TableHead';
import Typography from "@material-ui/core/Typography"
import IconButton from "@material-ui/core/IconButton"
import Divider from '@material-ui/core/Divider';
import Button from '@material-ui/core/Button';
import LinearProgress from '@material-ui/core/LinearProgress';

import MainContent from "components/ConsoleContents/MainContent"
import ContentBar from "components/ConsoleContents/ContentBar"

import FileDropzone from "components/FileUpload/FileDropzone"
import UploadProgressBar from 'components/FileUpload/UploadProgressBar';
import ForkbaseStatus from "components/ConsoleContents/ForkbaseStatus"

// read query-string
import queryString from 'query-string'

const styles = theme => ({
  block: {
    display: "block",
  },
  addDS: {
    marginRight: theme.spacing(1),
  },
  contentWrapper: {
    margin: "16px 16px",
    //position: "relative",
    minHeight: 200,
  },
  // for query-params
  pos: {
    marginBottom: 12,
  },
})

class RunPrediction extends React.Component {
  static propTypes = {
    classes: PropTypes.object.isRequired,
    handleHeaderTitleChange: PropTypes.func,
    resetLoadingBar: PropTypes.func,
  }

  state = {
    app:"",
    appVersion:"",
    predictorHost:"",
    FormIsValid: false,
    noPredictorSelected: false,
    // for file upload
    // populate the files state from FileDropzone
    selectedFiles: [],
    // post-upload status:
    message: "",
    uploadPercentage: 0,
    formState: "init",
  }

  componentDidMount() {
    this.props.handleHeaderTitleChange("Inference Jobs > Run Prediction")
    // read the query string from URL
    const values = queryString.parse(this.props.location.search)
    console.log("queryString parse: ", values)
    if (values.app && values.appVersion && values.predictorHost) {
      this.setState({
        app: values.app,
        appVersion: values.appVersion,
        predictorHost: values.predictorHost,
      })
    } else {
      this.setState({
        noPredictorSelected: true
      })
    }
  }

  componentDidUpdate(prevProps, prevState) {
    // if form's states have changed
    if (
      this.state.selectedFiles !== prevState.selectedFiles
    ) {
      if (
        this.state.app &&
        this.state.appVersion &&
        this.state.selectedFiles.length !== 0 &&
        this.state.predictorHost
      ) {
        this.setState({
          FormIsValid: true
        })
      // otherwise disable COMMIT button
      } else {
        this.setState({
          FormIsValid: false
        })
      }
    }
  }

  handleCommit = async e => {
    e.preventDefault();
    // reset the right-hand side
    // reset the ForkBase Status field:
    // this.props.resetResponses()
    // first reset COMMIT disabled
    this.setState({
      uploadPercentage: 0,
      FormIsValid: false,
      // set formState to loading
      formState: "loading",
    })

    // construct form data for sending
    // FormData() is default native JS object
    const formData = new FormData()
    // append(<whatever name>, value, <namePropterty>)
    console.log("selectedFiles[0]: ", this.state.selectedFiles[0])
    // flask createDS endpoint will look for
    // 'dataset' in request.files
    formData.append("img", this.state.selectedFiles[0])

    try {
      const res = await axios.post(
        `http://${this.state.predictorHost}/predict`,
        formData, 
        {
          headers: {
            'Content-Type': 'multipart/form-data',
            //"Authorization": `Bearer ${this.props.reduxToken}`
          },
          onUploadProgress: progressEvent => {
            // progressEvent will contain loaded and total
            let percentCompleted = parseInt(
              Math.round( (progressEvent.loaded * 100) / progressEvent.total )
            )
            console.log("From EventEmiiter, file Uploaded: ", percentCompleted)
            this.setState({
              uploadPercentage: percentCompleted
            })
          }
        }
      );
      // res.data is the object sent back from the server
      console.log("file uploaded, axios res.data: ", res.data)
      console.log("axios full response schema: ", res)

      this.setState(prevState => ({
        formState: "idle",
        message: "Upload and prediction done"
      }))
    } catch (err) {
      console.error(err, "error")
      this.setState({
        message: "Upload failed"
      })
    }
  }

  componentWillUnmount() {
    this.props.resetLoadingBar()
  }

  onDrop = files => {
    // file input, can access the file props
    // files is an array
    // files[0] is the 1st file we added
    // console.log(event.target.files[0])
    console.log("onDrop called, acceptedFiles: ", files)
    this.setState({
      selectedFiles: files
    })
  }

  handleRemoveCSV = () => {
    this.setState({
      selectedFiles: []
    })
    console.log("file removed")
  }

  render() {
    const { classes } = this.props

    if (this.state.noPredictorSelected) {
      return (
        <MainContent>
          <ContentBar
            needToList={false}
            barTitle="Run Prediction"
          />
          <div className={classes.contentWrapper}>
            Please select a predictor from an inference job
          </div>
        </MainContent>
      )
    }

    return (
      <React.Fragment>
        <MainContent>
          <ContentBar
            needToList={false}
            barTitle="Run Prediction"
          />
          <div className={classes.contentWrapper}>
            <Typography gutterBottom>
              App Name: {this.state.app}
            </Typography>
            <Typography gutterBottom>
              App Version: {this.state.appVersion}
            </Typography>
            <Typography className={classes.pos}>
              Predictor Host: {this.state.predictorHost}
            </Typography>
            <Divider />
            <br />
            <Typography variant="h5" gutterBottom align="center">
              Upload Test Image
            </Typography>
            <FileDropzone
              files={this.state.selectedFiles}
              onCsvDrop={this.onDrop}
              onRemoveCSV={this.handleRemoveCSV}
              AcceptedMIMEtypes={`
                image/jpeg,
                image/jpg,
                image/png
              `}
              MIMEhelperText={`
              (Only image format will be accepted)
              `}
              UploadType={`Image`}
            />
            <br />
            <Button
              variant="contained"
              color="primary"
              onClick={this.handleCommit}
              disabled={
                !this.state.FormIsValid ||
                this.state.formState === "loading"}
            >
              Predict
            </Button>
            <ForkbaseStatus
              formState={this.state.formState}
            >
              {this.state.formState === "loading" &&
                <React.Fragment>
                  <LinearProgress color="secondary" />
                  <br />
                </React.Fragment>
              }
              <UploadProgressBar
                percentCompleted={this.state.uploadPercentage}
                fileName={
                  this.state.selectedFiles.length !== 0
                  ? this.state.selectedFiles[0]["name"]
                  : ""
                }
                formState={this.state.formState}
                dataset={this.state.newDataset}
              />
              <br />
              <Typography component="p">
                <b>{this.state.message}</b>
                <br />
              </Typography>
            </ForkbaseStatus>
          </div>
        </MainContent>
      </React.Fragment>
    )
  }
}

// const mapStateToProps = state => ({
//   InferenceJobsList: state.InferenceJobsReducer.InferenceJobsList,
// })

const mapDispatchToProps = {
  handleHeaderTitleChange: ConsoleActions.handleHeaderTitleChange,
  resetLoadingBar: ConsoleActions.resetLoadingBar,
  push: push,
}

export default compose(
  connect(null, mapDispatchToProps),
  withStyles(styles)
)(RunPrediction)
