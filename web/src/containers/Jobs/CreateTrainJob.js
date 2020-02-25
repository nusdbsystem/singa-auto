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

import CreateTrainJobForm from "components/ConsoleForms/CreateTrainJobForm"

// form fields components
import AppName from "components/ConsoleContents/AppName"
import ForkbaseStatus from "components/ConsoleContents/ForkbaseStatus"

// RegExp rules
import { validDsAndBranch } from "regexp-rules";

const styles = theme => ({
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
  }

  componentDidMount() {
    this.props.handleHeaderTitleChange("Training Jobs > Create Train Job")
    this.props.requestListDS()
    this.props.requestAvailableModelList()
  }

  componentWillUnmount() {
    this.props.resetLoadingBar()
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

  render() {
    const { classes, DatasetsList, AvailableModelList } = this.props

    return (
      <MainContent>
        <ContentBar
          needToList={false}
          barTitle="Create Train Job"
        />
        <div className={classes.contentWrapper}>
          <Grid container spacing={6}>
            <Grid item xs={6}>
              <AppName
                title="1. Application Name"
                newAppName={this.state.newAppName}
                onHandleChange={this.handleChange}
                isCorrectInput={this.state.validDsName}
              />
              <br />
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
                <br />
                <Typography component="p">
                  <b>{this.state.message}</b>
                  <br />
                </Typography>
              </ForkbaseStatus>
            </Grid>
          </Grid>
        </div>
        <CreateTrainJobForm
            datasets={DatasetsList}
            models={AvailableModelList}
            postCreateTrainJob={this.props.postCreateTrainJob}
          />
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
