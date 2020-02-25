import React from "react"

import { withStyles } from "@material-ui/core/styles"
import { compose } from "redux"
import { connect } from "react-redux"

import { Link } from "react-router-dom"

import { goBack } from "connected-react-router"

// Material UI
import { Typography, Grid, Button } from "@material-ui/core"

import * as ConsoleActions from "../ConsoleAppFrame/actions"
import * as actions from "./actions"

// Import Layout
import MainContent from "components/ConsoleContents/MainContent"
import ContentBar from "components/ConsoleContents/ContentBar"

const styles = theme => ({
  contentWrapper: {
    margin: "40px 16px",
    //position: "relative",
  },
})

class CreateInferenceJob extends React.Component {
  componentDidMount() {}

  onClick = () => {
    const { app, appVersion } = this.props.match.params
    const budget = { GPU_COUNT: 0 }
    this.props.postCreateInferenceJob(app, appVersion, budget) // action.json
  }

  render() {
    const { classes } = this.props

    const { appId, app, appVersion } = this.props.match.params

    return (
      <React.Fragment>
        <MainContent>
          <ContentBar needToList={false} barTitle="Create Train Jobs" />
          <Grid container spacing={10} justify="center" alignItems="center">
            <Grid item>
              <div className={classes.contentWrapper}>
                <Typography>
                  Are you sure want to create an inference job for app: {app} |
                  appVersion: {appVersion}
                </Typography>
              </div>
            </Grid>
          </Grid>
          <Grid
            container
            spacing={5}
            justify="center"
            alignItems="center"
            style={{ minHeight: "100px" }}
          >
            <Grid item>
              <Button
                onClick={this.onClick}
                color="primary"
                variant="contained"
              >
                Create Inference Job
              </Button>
            </Grid>
            <Grid item>
              <Link to={`/console/jobs/trials/${appId}/${app}/${appVersion}`}>
                <Button
                  color="default"
                  variant="contained"
                  onClick={this.props.goBack}
                >
                  Go Back
                </Button>
              </Link>
            </Grid>
          </Grid>
        </MainContent>
      </React.Fragment>
    )
  }
}

const mapStateToProps = state => ({
  location: state.router.location,
})

const mapDispatchToProps = {
  handleHeaderTitleChange: ConsoleActions.handleHeaderTitleChange,
  postCreateInferenceJob: actions.postCreateInferenceJob,
  resetLoadingBar: ConsoleActions.resetLoadingBar,
  goBack: goBack,
}

export default compose(
  connect(mapStateToProps, mapDispatchToProps),
  withStyles(styles)
)(CreateInferenceJob)
