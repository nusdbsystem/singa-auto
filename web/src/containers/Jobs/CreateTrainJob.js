import React from "react"

import { compose } from "redux"
import { connect } from "react-redux"

// Material UI
import { withStyles } from "@material-ui/core/styles"
import Typography from '@material-ui/core/Typography';
import Grid from '@material-ui/core/Grid';
import Button from '@material-ui/core/Button';
import LinearProgress from '@material-ui/core/LinearProgress';

import * as ModelActions from "../Models/actions"
import * as DatasetActions from "../Datasets/actions"
import * as ConsoleActions from "../ConsoleAppFrame/actions"
import * as actions from "./actions"

// Import Layout
import MainContent from "components/ConsoleContents/MainContent"
import ContentBar from "components/ConsoleContents/ContentBar"

// form fields components
import AppName from "components/ConsoleContents/AppName"
import TaskName from "components/ConsoleContents/TaskName"
import DatasetSelect from "components/ConsoleContents/DatasetSelect"
import ModelSelect from "components/ConsoleContents/ModelSelect"
import BudgetInputs from "components/ConsoleContents/BudgetInputs"
import ForkbaseStatus from "components/ConsoleContents/ForkbaseStatus"

// RegExp rules
import { validDsAndBranch } from "regexp-rules";

const styles = theme => ({
  paper: {
    maxWidth: 700,
    margin: "auto",
    overflow: "hidden",
    marginBottom: 20,
    position: "relative",
    paddingBottom: 80,
  },
  contentWrapper: {
    margin: "40px 16px",
    //position: "relative",
    minHeight: 200,
  },
})

class CreateTrainJob extends React.Component {
  state = {
    newAppName: "",
    validDsName: true,
    FormIsValid: false,
    formState: "init",
    // formState => init | loading | idle
    message: "",
    task: "IMAGE_CLASSIFICATION",
    selectedTrainingDS: "",
    selectedValidationDS: "",
    selectedModel: "",
    Budget_TIME_HOURS: 0.1,
    Budget_GPU_COUNT: 0,
    Budget_MODEL_TRIAL_COUNT: -1,
  }

  componentDidMount() {
    this.props.handleHeaderTitleChange("Training Jobs > Create Train Job")
    this.props.requestListDS()
    this.props.requestAvailableModelList()
  }

  componentWillUnmount() {
    this.props.resetLoadingBar()
  }

  componentDidUpdate(prevProps, prevState) {
    // if form's states have changed
    if (
      this.state.newAppName !== prevState.newAppName ||
      this.state.selectedTrainingDS !== prevState.selectedTrainingDS ||
      this.state.selectedValidationDS !== prevState.selectedValidationDS ||
      this.state.selectedModel !== prevState.selectedModel ||
      this.state.Budget_GPU_COUNT !== prevState.Budget_GPU_COUNT ||
      this.state.Budget_TIME_HOURS !== prevState.Budget_TIME_HOURS ||
      this.state.Budget_MODEL_TRIAL_COUNT !== prevState.Budget_MODEL_TRIAL_COUNT
    ) {
      if (
        this.state.newAppName &&
        this.state.validDsName &&
        this.state.task &&
        this.state.selectedTrainingDS &&
        this.state.selectedValidationDS &&
        this.state.selectedModel &&
        this.state.Budget_TIME_HOURS !== "" &&
        this.state.Budget_GPU_COUNT !== "" &&
        this.state.Budget_MODEL_TRIAL_COUNT !== ""
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

  handleChange = name => event => {
    console.log("handleCHANGE NAME: ", name)
    if (name === "newAppName") {
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

   handleCommit = () => {
    let createTrainJobJSON = {}
    createTrainJobJSON.app = this.state.newAppName
    console.log("***************createTrainJobJSON: ", createTrainJobJSON)
    const gatheredValue = {
      app: this.state.newAppName,
      task: this.state.task,
      train_dataset_id: this.state.selectedTrainingDS,
      val_dataset_id: this.state.selectedValidationDS,
      budget: {
        TIME_HOURS: parseFloat(this.state.Budget_TIME_HOURS),
        GPU_COUNT: parseInt(this.state.Budget_GPU_COUNT),
        MODEL_TRIAL_COUNT: parseInt(this.state.Budget_MODEL_TRIAL_COUNT),
      },
      model_ids: [this.state.selectedModel],
      train_args: {},
    }
    console.log("***************gatheredValue: ", gatheredValue)
    this.props.postCreateTrainJob(gatheredValue)
  }

  render() {
    const { classes, DatasetsList, AvailableModelList } = this.props

    // Options for datasets
    const datasetOptions = DatasetsList.map(dataset => {
      return {
        value: dataset.id,
        label: dataset.name + "(" + dataset.id + ")",
      }
    })

    // Options for models
    const modelOptions = AvailableModelList.map(model => {
      return {
        value: model.id,
        label: model.name + "(" + model.id + ")",
      }
    })

    console.log(">>>>>>CreateTrainJob State: ", this.state)

    return (
      <MainContent classes={{paper:classes.paper}}>
        <ContentBar
          needToList={false}
          barTitle="Create Train Job"
        />
        <div className={classes.contentWrapper}>
          <Grid container spacing={6}>
            <Grid item xs={12}>
              <AppName
                title="1. Application Name"
                newAppName={this.state.newAppName}
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
              <DatasetSelect
                title="3. Dataset for Training"
                purpose="Training"
                datasetList={datasetOptions}
                selectedDataset={this.state.selectedTrainingDS}
                onHandleChange={this.handleChange}
              />
              <br />
              <DatasetSelect
                title="4. Dataset for Validation"
                purpose="Validation"
                datasetList={datasetOptions}
                selectedDataset={this.state.selectedValidationDS}
                onHandleChange={this.handleChange}
              />
              <br />
              <BudgetInputs
                title="5. Budget"
                value_time_hours={this.state.Budget_TIME_HOURS}
                value_gpu_count={this.state.Budget_GPU_COUNT}
                value_model_trial_count={this.state.Budget_MODEL_TRIAL_COUNT}
                onHandleChange={this.handleChange}
              />
              <br />
              {/* TODO: UI notify user of recommended model */}
              <ModelSelect
                title="6. Model"
                modelList={modelOptions}
                selectedModel={this.state.selectedModel}
                onHandleChange={this.handleChange}
              />
              <br />
              <Grid
                container
                direction="row"
                justify="center"
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
            <Grid item xs={12}>
              <ForkbaseStatus
                formState={this.state.formState}
              >
                {this.state.formState === "loading" &&
                  <React.Fragment>
                    <LinearProgress color="secondary" />
                    <br />
                  </React.Fragment>
                }
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
  DatasetsList: state.DatasetsReducer.DatasetList,
  AvailableModelList: state.ModelsReducer.AvailableModelList,
})

const mapDispatchToProps = {
  handleHeaderTitleChange: ConsoleActions.handleHeaderTitleChange,
  postCreateTrainJob: actions.createTrainJob,
  requestListDS: DatasetActions.requestListDS,
  resetLoadingBar: ConsoleActions.resetLoadingBar,
  requestAvailableModelList: ModelActions.requestAvailableModelList,
}

export default compose(
  connect(mapStateToProps, mapDispatchToProps),
  withStyles(styles)
)(CreateTrainJob)
