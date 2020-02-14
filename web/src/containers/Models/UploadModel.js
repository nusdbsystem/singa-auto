import React from "react"
import PropTypes from "prop-types"
import axios from 'axios';
import HTTPconfig from "HTTPconfig"
import { connect } from "react-redux"
import { compose } from "redux"

import * as ConsoleActions from "../ConsoleAppFrame/actions"

import { withStyles } from "@material-ui/core/styles"
import Typography from '@material-ui/core/Typography';
import Grid from '@material-ui/core/Grid';
import Button from '@material-ui/core/Button';
import LinearProgress from '@material-ui/core/LinearProgress';

import FileDropzone from "components/FileUpload/FileDropzone"
import UploadProgressBar from 'components/FileUpload/UploadProgressBar';
import MainContent from "components/ConsoleContents/MainContent"
import ContentBar from "components/ConsoleContents/ContentBar"
import ModelName from "components/ConsoleContents/ModelName"
import TaskName from "components/ConsoleContents/TaskName"
import ForkbaseStatus from "components/ConsoleContents/ForkbaseStatus"
import ModelClassSelect from "components/ConsoleContents/ModelClassSelect"

// RegExp rules
import { validDsAndBranch } from "regexp-rules";

const styles = theme => ({
  contentWrapper: {
    // position: "relative",
    // display: "flex",
    // alignItems: "center",
    // justifyContent: "center"
    margin: "10px 16px",
    //position: "relative",
    minHeight: 200,
  },
})

class UploadModel extends React.Component {
  /**
   * a sample Model.py to upload is the
   * PyPandaVgg.py in the examples\models\image_classification
   * other sample Model to upload can be PyPandaDenseNet.py
   *
   * The other models have their own dependency too.
   * (should be designated by the model builder as well)
   * That is to say, the dependency is needed when uploading every new model.
   */
  state = {
    newModel:"",
    validDsName: true,
    FormIsValid: false,
    formState: "init",
    // formState => init | loading | idle
    // for file upload
    // populate the files state from FileDropzone
    selectedFiles: [],
    message: "",
    uploadPercentage: 0,
    task: "IMAGE_CLASSIFICATION",
    // model_class, for Feb 2020, use two sample model files
    // their model-classes are:
    modelClass: [
      "PyPandaVgg",
      "PyPandaDenseNet",
    ]
  }

  static propTypes = {
    classes: PropTypes.object.isRequired,
    handleHeaderTitleChange: PropTypes.func,
    resetLoadingBar: PropTypes.func,
    reduxToken: PropTypes.string.isRequired,
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

  handleChange = name => event => {
    if (name === "newModel") {
      if (
        validDsAndBranch.test(event.target.value) &&
        event.target.value.length <= 50
      ) {
        this.setState({
          validDsName: true
        });
      } else {
        this.setState({
          validDsName: false
        });
      }
    }
    // in future will have more tasks
    this.setState({
      [name]: event.target.value,
    });
  };

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
    formData.append("dataset", this.state.selectedFiles[0])
    formData.append("name", this.state.newModel)
    formData.append("task", this.state.task)

    try {
      const res = await axios.post(
        `${HTTPconfig.gateway}datasets`,
        formData, 
        {
          headers: {
            'Content-Type': 'multipart/form-data',
            "Authorization": `Bearer ${this.props.reduxToken}`
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
        message: "Upload success"
      }))
    } catch (err) {
      console.error(err, "error")
      this.setState({
        message: "upload failed"
      })
    }
  }

  componentDidMount() {
    this.props.handleHeaderTitleChange("Model > New Model")
  }

  componentDidUpdate(prevProps, prevState) {
    // if form's states have changed
    if (
      this.state.newModel !== prevState.newModel ||
      this.state.selectedFiles !== prevState.selectedFiles
    ) {
      if (
        this.state.newModel &&
        this.state.validDsName &&
        this.state.selectedFiles.length !== 0 &&
        this.state.task
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

  componentWillUnmount() {
    this.props.resetLoadingBar()
  }

  render() {
    const { classes } = this.props

    return (
      <MainContent>
        <ContentBar
          needToList={false}
          barTitle="Upload Model"
        />
        <div className={classes.contentWrapper}>
          <Grid container spacing={6}>
            <Grid item xs={6}>
              <ModelName
                title="1. Model Name"
                newModel={this.state.newModel}
                onHandleChange={this.handleChange}
                isCorrectInput={this.state.validDsName}
              />
              <br />
              <TaskName
                title="2. Task Name"
                task={this.state.task}
                onHandleChange={this.handleChange}
              />
              <br />
              <Typography variant="h5" gutterBottom align="center">
                3. Upload Model
              </Typography>
              <FileDropzone
                files={this.state.selectedFiles}
                onCsvDrop={this.onDrop}
                onRemoveCSV={this.handleRemoveCSV}
                AcceptedMIMEtypes={`
                  application/x-python-code,
                  text/x-python
                `}
                MIMEhelperText={`
                (Only *.py script files will be accepted)
                `}
                UploadType={`Model`}
              />
              <br />
              <ModelClassSelect
                title="4. Model Class"
                modelClass={this.state.modelClass}
                onHandleChange={this.handleChange}
              />
              <Grid
                container
                direction="row"
                justify="flex-end"
                alignItems="center"
              >
                <Button
                  variant="contained"
                  color="primary"
                  onClick={this.handleCommit}
                  disabled={
                    !this.state.FormIsValid ||
                    this.state.formState === "loading"}
                >
                  COMMIT
                </Button>
              </Grid>
            </Grid>
            <Grid item xs={6}>
              <ForkbaseStatus
                formState={this.state.formState}
              >
                {this.state.stylesformState === "loading" &&
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
                  dataset={this.state.newModel}
                />
                <br />
                <Typography component="p">
                  <b>{this.state.message}</b>
                  <br />
                </Typography>
              </ForkbaseStatus>
            </Grid>
          </Grid>
        </div>
      </MainContent>
    )
  }
}

const mapStateToProps = state => ({
  reduxToken: state.Root.token,
})

const mapDispatchToProps = {
  handleHeaderTitleChange: ConsoleActions.handleHeaderTitleChange,
  // postCreateDataset: actions.postCreateDataset,
  resetLoadingBar: ConsoleActions.resetLoadingBar,
}

export default compose(
  connect(mapStateToProps, mapDispatchToProps),
  withStyles(styles)
)(UploadModel)
