import React from "react"

import { withStyles } from "@material-ui/core/styles"
import { compose } from "redux"
import { connect } from "react-redux"

// temp use axios in containers level,
// TODO: use sagas and service in future
import axios from 'axios';
import HTTPconfig from "HTTPconfig"

import { goBack, push } from "connected-react-router"

// Material UI
import Typography from '@material-ui/core/Typography';
import Grid from '@material-ui/core/Grid';
import Button from '@material-ui/core/Button';
import Table from '@material-ui/core/Table';
import TableBody from '@material-ui/core/TableBody';
import TableCell from '@material-ui/core/TableCell';
import TableHead from '@material-ui/core/TableHead';
import TableRow from '@material-ui/core/TableRow';

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
  state = {
    selectedInferenceJob: {},
    message: "N.A.",
  }

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

  handleClickStopInferenceJob = async () => {
    const { app, appVersion } = this.props.match.params
    const { token } = this.props
    try {
      const res = await axios({
        method: "post",
        url: `${HTTPconfig.gateway}inference_jobs/${app}/${appVersion}/stop`,
        headers: {
          "Authorization": `Bearer ${token}`
        },
      });

      // res.data is the object sent back from the server
      console.log("axios res.data: ", res.data)
      console.log("axios full response schema: ", res)

      this.setState(prevState => ({
        message: "Stop Inference Job Success"
      }))
    } catch (err) {
      console.error(err, "error")
      this.setState({
        message: "Failed to stop inference job"
      })
    }
  }

  handleClickRunPrediction = () => {
    const { app, appVersion } = this.props.match.params
    const x = this.state.selectedInferenceJob

    const url = (`/console/inferencejobs/run-prediction` +
      `?app=${app}` +
      `&appVersion=${appVersion}` +
      `&predictorHost=${x.predictor_host}`)

    console.log("redirect url: ", url)
    this.props.push(url)
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
                    <TableCell>Predictor Host</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {x ? (
                    <TableRow key={x.id} hover>
                      <TableCell>{x.id}</TableCell>
                      <TableCell>{x.app}</TableCell>
                      <TableCell>{x.app_version}</TableCell>
                      <TableCell>
                        {x.datetime_started
                          ? moment(x.datetime_started).fromNow()
                          : "Stopped"}
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
            {this.state.selectedInferenceJob.datetime_started &&
              (
                <>
                  <Grid item >
                    <Button
                      onClick={this.handleClickRunPrediction}
                      color="secondary"
                      variant="contained"
                    >
                      Predict
                    </Button>
                  </Grid>
                  <Grid item>
                    <Button
                      onClick={this.handleClickStopInferenceJob}
                      color="primary"
                      variant="contained"
                    >
                      Stop Inference Job
                    </Button>
                  </Grid>
                </>
              )
            }
            <Grid item>
              <Button
                color="default"
                variant="contained"
                onClick={this.props.goBack}
              >
                Go Back
              </Button>
            </Grid>
            <Grid
              container
              spacing={5}
              justify="center"
              alignItems="center"
              style={{ minHeight: "100px" }}
            >
            <Typography component="p">
              System Message: <b>{this.state.message}</b>
            </Typography>
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
  push: push,
}

export default compose(
  connect(mapStateToProps, mapDispatchToProps),
  withStyles(styles)
)(InferenceJobDetails)
