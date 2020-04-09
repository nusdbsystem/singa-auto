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

// for checkboxes of dependencies
import FormLabel from '@material-ui/core/FormLabel';
import FormControl from '@material-ui/core/FormControl';
import FormGroup from '@material-ui/core/FormGroup';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import FormHelperText from '@material-ui/core/FormHelperText';
import Checkbox from '@material-ui/core/Checkbox';

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
    margin: "40px 16px",
    //position: "relative",
    minHeight: 200,
  },
  // for checkboxes of dependencies
  root: {
    display: 'flex',
  },
  formControl: {
    margin: theme.spacing(3),
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
    ],
    selectedModelClass: "PyPandaVgg",
    // dependencies
    torch101: false,
    torchvision022: false,
    matplotlib310: false,
    lime01136: false,
    scikitLearn0200: false,
    tensorflow1120: false,
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
    // console.log("handleCHANGE NAME: ", name)
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

  hancleCheckboxClick = name => event => {
    this.setState({
      [name]: event.target.checked
    });
  };

  handleCommit = async e => {
    // console.log("THIS STATE: ", this.state)
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
    // console.log("selectedFiles[0]: ", this.state.selectedFiles[0])
    // flask /models (create_model in app.py) endpoint will look for
    // 'model_file_bytes' in request.files
    formData.append("model_file_bytes", this.state.selectedFiles[0])
    formData.append("name", this.state.newModel)
    formData.append("task", this.state.task)
    formData.append("model_class", this.state.selectedModelClass)
    // flask /models (create_model in app.py) endpoint also
    // checks for dependencies
    let depState = {}
    depState["torch101"] = this.state.torch101
    depState["torchvision022"] = this.state.torchvision022
    depState["matplotlib310"] = this.state.matplotlib310
    depState["lime01136"] = this.state.lime01136
    depState["scikitLearn0200"] = this.state.scikitLearn0200
    depState["tensorflow1120"] = this.state.tensorflow1120
    //console.log("depState: ", depState)

    const checkedDep = Object.keys(depState)
      .filter(key => depState[key])

    //console.log("checkedDep: ", checkedDep)

    let depToPOST = {}
    checkedDep.map(v => {
      switch(v) {
        case "torch101":
          depToPOST["torch"] = "1.0.1"
          return "torch101"
        case "torchvision022":
          depToPOST["torchvision"] = "0.2.2"
          return "torchvision022"
        case "matplotlib310":
          depToPOST["matplotlib"] = "3.1.0"
          return "matplotlib310"
        case "lime01136":
          depToPOST["lime"] = "0.1.1.36"
          return "lime01136"
        case "scikitLearn0200":
          depToPOST["scikit-learn"] = "0.20.0"
          return "scikitLearn0200"
        case "tensorflow1120":
          depToPOST["tensorflow"] = "1.12.0"
          return "tensorflow1120"
        default:
          return "NA"
      }
    })
    // console.log("depTOPOST stringified: ", JSON.stringify(depToPOST))
    formData.append("dependencies", JSON.stringify(depToPOST))

    try {
      const res = await axios.post(
        `${HTTPconfig.gateway}models`,
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
      this.state.selectedFiles !== prevState.selectedFiles ||
      // make sure dependencies selected
      this.state.torch101 !== prevState.torch101 ||
      this.state.torchvision022 !== prevState.torchvision022 ||
      this.state.matplotlib310 !== prevState.matplotlib310 ||
      this.state.lime01136 !== prevState.lime01136 ||
      this.state.scikitLearn0200 !== prevState.scikitLearn0200 ||
      this.state.tensorflow1120 !== prevState.tensorflow1120
    ) {
      const noDependencies = [
        this.state.torch101,
        this.state.torchvision022,
        this.state.matplotlib310,
        this.state.lime01136,
        this.state.scikitLearn0200,
        this.state.tensorflow1120
      ].filter(v => v).length === 0;

      if (
        this.state.newModel &&
        this.state.validDsName &&
        this.state.selectedFiles.length !== 0 &&
        this.state.task &&
        !noDependencies
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

    const {
      torch101,
      torchvision022,
      matplotlib310,
      lime01136,
      scikitLearn0200,
      tensorflow1120
    } = this.state

    const error = [
      torch101,
      torchvision022,
      matplotlib310,
      lime01136,
      scikitLearn0200,
      tensorflow1120
    ].filter(v => v).length === 0;

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
                selectedModelClass={this.state.selectedModelClass}
                onHandleChange={this.handleChange}
              />
              <br />
              <Typography variant="h5" gutterBottom align="center">
                5. Dependencies
              </Typography>
              <Grid
                container
                direction="row"
                justify="space-evenly"
                alignItems="center"
              >
                <Grid item>
                  <div className={classes.root}>
                    <FormControl required error={error} component="fieldset" className={classes.formControl}>
                      <FormLabel component="legend">
                        Required
                      </FormLabel>
                      <FormGroup>
                        <FormControlLabel
                          control={<Checkbox checked={torch101} onChange={this.hancleCheckboxClick('torch101')} value="torch101" />}
                          label="torch 1.0.1"
                        />
                        <FormControlLabel
                          control={<Checkbox checked={torchvision022} onChange={this.hancleCheckboxClick('torchvision022')} value="torchvision022" />}
                          label="torchvision 0.2.2"
                        />
                        <FormControlLabel
                          control={<Checkbox checked={matplotlib310} onChange={this.hancleCheckboxClick('matplotlib310')} value="matplotlib310" />}
                          label="matplotlib 3.1.0"
                        />
                        <FormControlLabel
                          control={<Checkbox checked={lime01136} onChange={this.hancleCheckboxClick('lime01136')} value="lime01136" />}
                          label="lime 0.1.1.36"
                        />
                        <FormControlLabel
                          control={<Checkbox checked={scikitLearn0200} onChange={this.hancleCheckboxClick('scikitLearn0200')} value="scikitLearn0200" />}
                          label="scikit-learn 0.20.0"
                        />
                        <FormControlLabel
                          control={
                            <Checkbox checked={tensorflow1120} onChange={this.hancleCheckboxClick('tensorflow1120')} value="tensorflow1120" />
                          }
                          label="tensorflow 1.12.0"
                        />
                      </FormGroup>
                      <FormHelperText>Can choose multiple</FormHelperText>
                    </FormControl>
                  </div>
                </Grid>
              </Grid>
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
