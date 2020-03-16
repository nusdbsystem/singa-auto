import React from "react"
import PropTypes from "prop-types"
import { connect } from "react-redux"
import { compose } from "redux"
import { push } from "connected-react-router"
import PageviewIcon from "@material-ui/icons/Pageview"

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

import MainContent from "components/ConsoleContents/MainContent"
import ContentBar from "components/ConsoleContents/ContentBar"

import FileDropzone from "components/FileUpload/FileDropzone"

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

  componentDidUpdate(prevProps, prevState) {}

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
