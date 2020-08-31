import React from "react"
import PropTypes from "prop-types"

import axios from 'axios';

import { connect } from "react-redux"
import { compose } from "redux"
import { push } from "connected-react-router"

import * as ConsoleActions from "../ConsoleAppFrame/actions"

import { withStyles } from "@material-ui/core/styles"
import Typography from "@material-ui/core/Typography"
import Divider from '@material-ui/core/Divider';
// import Button from '@material-ui/core/Button';
import LinearProgress from '@material-ui/core/LinearProgress';

// for display of response
import Grid from '@material-ui/core/Grid';

import MainContent from "components/ConsoleContents/MainContent"
import ContentBar from "components/ConsoleContents/ContentBar"

import FileDropzone from "components/FileUpload/FileDropzone"
import UploadProgressBar from 'components/FileUpload/UploadProgressBar';
import ForkbaseStatus from "components/ConsoleContents/ForkbaseStatus"

// read query-string
import queryString from 'query-string'

// for echart plot
import ReactEcharts from 'echarts-for-react';
import { calculateGaussian } from "./calculateGaussian"

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
  // for response display
  response: {
    flexGrow: 1,
    marginTop: "20px",
  },
  explainImg: {
    margin: "0 auto",
    width: "90%",
  }
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
    // populate the response
    predictionDone: false,
    gradcamImg: "",
    limeImg: "",
    mcDropout: [],
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
      // reset previous response, if any
      predictionDone: false,
      gradcamImg: "",
      limeImg: "",
      mcDropout: [],
      // upload
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
        `http://${this.state.predictorHost}`,
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
        message: "Upload and prediction done",
        predictionDone: true,
        gradcamImg: res.data.explanations.gradcam_img,
        limeImg: res.data.explanations.lime_img,
        mcDropout: res.data.mc_dropout,
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

  getOption = (mcDropout) => {
    console.log("mcDropout: ", mcDropout)

    return {
      title: {
        text: "MC Dropout",
        // x: "center"
      },
      // toolbox: {
      //   feature: {
      //     dataView: { show: true, readOnly: false },
      //     magicType: { show: true, type: ['line', 'bar'] },
      //     restore: { show: true },
      //     saveAsImage: { show: true }
      //   }
      // },
      legend: {
        data: mcDropout.map(item => item.label)
      },
      tooltip: {
        trigger: 'axis',
      },
      xAxis: {
        type: 'value',
        name: "Mean",
        nameLocation: 'middle',
        min: 0,
        max: 1
      },
      yAxis: {
        type: 'value',
        name: "Probability",
        min: 0,
        max: 1
      },
      series: mcDropout.map(item => {
        return {
          name: item.label,
          type: "line",
          data: calculateGaussian(item.mean, item.std)
        }
      })
    }
  };

  render() {
    console.log("STATE: ", this.state)
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
            {/* <Button
              variant="contained"
              color="primary"
              onClick={this.handleCommit}
              disabled={
                !this.state.FormIsValid ||
                this.state.formState === "loading"}
            >
              Predict
            </Button> */}
          </div>
        </MainContent>
        <MainContent>
          <ContentBar
            needToList={false}
            barTitle="Results"
          />
          <div className={classes.contentWrapper}>
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
            <br />
            {this.state.predictionDone &&
              <div className={classes.response}>
                <Grid container spacing={3}>
                  <Grid item xs={12} sm={6}>
                    <Typography variant="h5" gutterBottom align="center">
                      Gradcam Image:
                    </Typography>
                    <img
                      className={classes.explainImg}
                      src={`data:image/jpeg;base64,${this.state.gradcamImg}`}
                      alt="GradcamImg"
                    />
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <Typography variant="h5" gutterBottom align="center">
                      Lime Image:
                    </Typography>
                    <img
                      className={classes.explainImg}
                      src={`data:image/jpeg;base64,${this.state.limeImg}`}
                      alt="LimeImg"
                    />
                  </Grid>
                </Grid>
                <br />
                <Divider />
                <br />
                <ReactEcharts
                  option={this.getOption(this.state.mcDropout)}
                  style={{ height: 500 }}
                />
              </div>
            }
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
