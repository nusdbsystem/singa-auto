import React from "react"

import { withStyles } from "@material-ui/core/styles"
import { compose } from "redux"
import { connect } from "react-redux"

import { goBack } from "connected-react-router"

// Material UI
import {
  Button,
  Table,
  Grid,
  TableHead,
  TableBody,
  TableRow,
  TableCell,
} from "@material-ui/core"

import * as ConsoleActions from "../ConsoleAppFrame/actions"
import * as ClientAPI from "services/ClientAPI"

// Third parts
import * as moment from "moment"

// Import Layout
import MainContent from "components/ConsoleContents/MainContent"
import ContentBar from "components/ConsoleContents/ContentBar"

const styles = theme => ({
  contentWrapper: {
    margin: "40px 16px 0px 16px",
    //position: "relative",
    textAlign: "center",
  },
})

class InferenceJobDetails extends React.Component {
  state = { selectedInferenceJob: {} }

  async componentDidMount() {
    this.props.handleHeaderTitleChange("Inference Jobs > List Inference Jobs > Inference Job Details")
    const { app, appVersion } = this.props.match.params
    const { token } = this.props
    const response = await ClientAPI.get_running_inference_jobs(
      app,
      appVersion,
      {},
      token
    )
    const inferenceJob = response.data
    this.setState({ selectedInferenceJob: inferenceJob })
  }

  render() {
    const { classes } = this.props

    const { app, appVersion } = this.props.match.params
    const x = this.state.selectedInferenceJob

    return (
      <React.Fragment>
        <MainContent>
          <ContentBar
            needToList={false}
            barTitle="Running Inference Job"
          />
          <Grid container spacing={10} justify="center" alignItems="center">
            <Grid item xs={12}>
              <div className={classes.contentWrapper}>
                <p>
                  Running inference job for <b>{app}</b> | appVersion:{" "}
                  <b>{appVersion}</b>
                </p>
              </div>
            </Grid>
            <Grid item>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Inference Job ID</TableCell>
                    <TableCell>App Name</TableCell>
                    <TableCell>App Version</TableCell>
                    <TableCell>Started</TableCell>
                    <TableCell>Prediction Host</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {x ? (
                    <TableRow key={x.id} hover>
                      <TableCell>{x.id}</TableCell>
                      <TableCell>{x.app}</TableCell>
                      <TableCell>{x.app_version}</TableCell>
                      <TableCell>
                        {moment(x.datetime_started).fromNow()}
                      </TableCell>
                      <TableCell>{x.predictor_host}</TableCell>
                    </TableRow>
                  ) : (
                    ""
                  )}
                </TableBody>
              </Table>
            </Grid>
          </Grid>
          <Grid
            container
            spacing={5}
            justify="center"
            alignItems="center"
            style={{ minHeight: "100px" }}
          >
            {/* <Grid item >
                            <Button onClick={this.onClick} color="primary" variant="contained">
                                Stop Inference Job
                            </Button>
                        </Grid> */}
            <Grid item>
              <Button
                color="default"
                variant="contained"
                onClick={this.props.goBack}
              >
                Go Back
              </Button>
            </Grid>
          </Grid>
        </MainContent>
      </React.Fragment>
    )
  }
}

const mapStateToProps = state => ({
  token: state.Root.token,
})

const mapDispatchToProps = {
  handleHeaderTitleChange: ConsoleActions.handleHeaderTitleChange,
  resetLoadingBar: ConsoleActions.resetLoadingBar,
  goBack: goBack,
}

export default compose(
  connect(mapStateToProps, mapDispatchToProps),
  withStyles(styles)
)(InferenceJobDetails)
