import React from "react"
import { connect } from "react-redux"
import { compose } from "redux"
import PropTypes from "prop-types"
import { Switch, Route, Redirect } from "react-router-dom"

import { ThemeProvider, withStyles } from "@material-ui/core/styles"
import Hidden from "@material-ui/core/Hidden"

import Header from "components/ConsoleHeader/Header"
import Navigator from "components/ConsoleSideBar/Navigator"
import theme from "./ConsoleTheme"

// Datasets Component
import ListDataSets from "../Datasets/ListDataSets"
import UploadDataset from "../Datasets/UploadDataset"

// Models Component
import ListAvailableModels from "../Models/ListAvailableModels"
import UploadModel from "../Models/UploadModel"

// Trainjobs Component
import ListTrainJobs from "../Jobs/ListTrainJobs"
import CreateTrainJob from "../Jobs/CreateTrainJob"
import ListTrials from "../Jobs/ListTrials"
import TrialDetails from "../Jobs/TrialsDetails"

// Inference Jobs Component
import InferenceJobDetails from "../InferenceJobs/InferenceJobDetails"
import ListInferenceJobs from "../InferenceJobs/ListInferenceJobs"
import CreateInferenceJob from "../InferenceJobs/CreateInferenceJob"

import Copyright from "components/ConsoleContents/Copyright"

import LoadingBar from "react-redux-loading-bar"

const drawerWidth = 256

const styles = {
  root: {
    display: "flex",
    minHeight: "100vh",
  },
  drawer: {
    [theme.breakpoints.up("sm")]: {
      width: drawerWidth,
      flexShrink: 0,
    },
  },
  app: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
  },
  main: {
    flex: 1,
    padding: theme.spacing(6, 4),
    background: "#eaeff1", // light grey
  },
  footer: {
    padding: theme.spacing(2),
    background: "#eaeff1",
  },
}

class ConsoleAppFrame extends React.Component {
  static propTypes = {
    authStatus: PropTypes.bool.isRequired,
    classes: PropTypes.object.isRequired,
    headerTitle: PropTypes.string,
  }

  state = {
    mobileOpen: false,
  }

  handleDrawerToggle = () => {
    // must use prevState
    this.setState(prevState => ({
      mobileOpen: !prevState.mobileOpen,
    }))
  }

  render() {
    const { authStatus, classes, headerTitle } = this.props

    if (!authStatus) {
      return <Redirect to="/sign-in" />
    }

    return (
      <ThemeProvider theme={theme}>
        <LoadingBar
          // only display if the action took longer than updateTime to finish
          // default updateTime = 200ms
          updateTime={300}
          progressIncrease={10}
          style={{
            backgroundColor: "#fc6e43",
            height: 8,
            zIndex: 2000,
            position: "fixed",
            top: 0,
          }}
        />
        <div className={classes.root}>
          <nav className={classes.drawer}>
            <Hidden smUp implementation="js">
              <Navigator
                PaperProps={{ style: { width: drawerWidth } }}
                variant="temporary"
                open={this.state.mobileOpen}
                onClose={this.handleDrawerToggle}
              />
            </Hidden>
            <Hidden xsDown implementation="css">
              <Navigator PaperProps={{ style: { width: drawerWidth } }} />
            </Hidden>
          </nav>
          <div className={classes.app}>
            <Header
              onDrawerToggle={this.handleDrawerToggle}
              title={headerTitle}
            />
            <main className={classes.main}>
              <Switch>
                {/* ***************************************
                  * Datasets
                  * ***************************************/}
                <Route
                  exact
                  path="/console/datasets/list-datasets"
                  component={ListDataSets}
                />
                <Route
                  exact
                  path="/console/datasets/upload-dataset"
                  component={UploadDataset}
                />
                {/* ***************************************
                  * Models
                  * ***************************************/}
                <Route
                  exact
                  path="/console/models/list-models"
                  component={ListAvailableModels}
                />
                <Route
                  exact
                  path="/console/models/upload-model"
                  component={UploadModel}
                />
                {/* ***************************************
                  * Train Jobs
                  * ***************************************/}
                <Route
                  exact
                  path="/console/jobs/list-train-jobs"
                  component={ListTrainJobs}
                />
                <Route
                  exact
                  path="/console/jobs/create-train-job"
                  component={CreateTrainJob}
                />
                {/* ***************************************
                  * Trials
                  * ***************************************/}
                <Route
                  exact
                  path="/console/jobs/trials/:appId/:app/:appVersion"
                  component={ListTrials}
                />
                <Route
                  exact
                  path="/console/jobs/trials/:trialId"
                  component={TrialDetails}
                />
                {/* ***************************************
                  * Inference Jobs
                  * ***************************************/}
                <Route
                  exact
                  path="/console/inferencejobs/:appId/:app/:appVersion/create_inference_job"
                  component={CreateInferenceJob}
                />
                <Route
                  exact
                  path="/console/inferencejobs/list-inferencejobs"
                  component={ListInferenceJobs}
                />
                <Route
                  exact
                  path="/console/inferencejobs/running_job/:app/:appVersion"
                  component={InferenceJobDetails}
                />
              </Switch>
            </main>
            <footer className={classes.footer}>
              <Copyright />
            </footer>
          </div>
        </div>
      </ThemeProvider>
    )
  }
}

const mapStateToProps = state => ({
  authStatus: !!state.Root.token,
  headerTitle: state.ConsoleAppFrame.headerTitle,
})

export default compose(
  connect(mapStateToProps),
  withStyles(styles)
)(ConsoleAppFrame)
