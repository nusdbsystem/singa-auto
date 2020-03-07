/*
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at
 
    http://www.apache.org/licenses/LICENSE-2.0
 
  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
 */

import React from "react"
import { connect } from "react-redux"
import { compose } from "redux"
import { withRouter } from "react-router-dom"

import * as moment from "moment"

import PlotManager from "app/PlotManager"
import RafikiClient from "app/RafikiClient"
import getPlotDetails from "app/getPlotDetails"

import * as ConsoleActions from "../ConsoleAppFrame/actions"

import HTTPconfig from "HTTPconfig"

import { withStyles } from '@material-ui/core/styles';
import Paper from '@material-ui/core/Paper';
import List from '@material-ui/core/List';
import ListItem from '@material-ui/core/ListItem'
import ListItemText from '@material-ui/core/ListItemText';
import Table from '@material-ui/core/Table';
import TableBody from '@material-ui/core/TableBody';
import TableCell from '@material-ui/core/TableCell';
import TableRow from '@material-ui/core/TableRow'
import Typography from '@material-ui/core/Typography';
import Divider from '@material-ui/core/Divider';
import Button from '@material-ui/core/Button';
import LinearProgress from '@material-ui/core/LinearProgress';


const styles = theme => ({
  headerSub: {
    fontSize: theme.typography.h4.fontSize,
    margin: theme.spacing(2),
  },
  detailsPaper: {
    margin: theme.spacing(2),
  },
  messagesPaper: {
    margin: theme.spacing(2),
  },
  plotPaper: {
    width: "100%",
    maxWidth: 800,
    height: 500,
    padding: theme.spacing(1),
    paddingTop: theme.spacing(2),
    margin: theme.spacing(4),
  },
  divider: {
    margin: theme.spacing(4),
  },
  mainBtn: {
    marginLeft: theme.spacing(5),
  },
})


class TrialDetails extends React.Component {
  constructor(props) {
    super(props)
    this.state = { logs: null, trial: null }
    this.chart = []
    const adminHost = HTTPconfig.adminHost || "localhost"
    const adminPort = HTTPconfig.adminPort || 3000
    console.log("adminHost: ", adminHost)
    console.log("HTTPconfig.adminHost: ", HTTPconfig.adminHost)
    console.log("adminPort: ", adminPort)
    this.rafikiClient = new RafikiClient(adminHost, adminPort)
    this.plotManager = new PlotManager()
  }

  async componentDidMount() {
    const { trialId } = this.props.match.params
    this.props.handleHeaderTitleChange("Training Jobs > Jobs List > List Trials > Trial Details")

    try {
      const [logs, trial] = await Promise.all([
        this.rafikiClient.getTrialLogs(trialId),
        this.rafikiClient.getTrial(trialId),
      ])
      this.setState({ logs, trial })
    } catch (error) {
      console.log(error, "Failed to retrieve trial & its logs")
    }
  }

  componentWillUnmount() {
    this.props.resetLoadingBar()
  }

  componentDidUpdate() {
    // TODO: how can this plot??
    this.updatePlots()
  }

  updatePlots() {
    const { logs } = this.state
    const plotManager = this.plotManager

    if (!logs) return

    for (const i in logs.plots) {
      const { series, plotOption } = getPlotDetails(logs.plots[i], logs.metrics)
      plotManager.updatePlot(`plot-${i}`, series, plotOption)
    }
  }

  renderDetails() {
    const { classes } = this.props
    const { trial } = this.state

    return (
      <React.Fragment>
        <Typography gutterBottom variant="h3">
          Details
        </Typography>
        <Paper className={classes.detailsPaper}>
          <Table>
            <TableBody>
              <TableRow>
                <TableCell>Trial ID</TableCell>
                <TableCell>{trial.id}</TableCell>
              </TableRow>
              <TableRow>
                <TableCell>Model</TableCell>
                <TableCell>{trial.model_name}</TableCell>
              </TableRow>
              <TableRow>
                <TableCell>Status</TableCell>
                <TableCell>{trial.status}</TableCell>
              </TableRow>
              {trial.score !== null && (
                <TableRow>
                  <TableCell>Score</TableCell>
                  <TableCell>{parseFloat(trial.score).toFixed(3)}</TableCell>
                </TableRow>
              )}
              {trial.proposal && (
                <TableRow>
                  <TableCell>Proposal</TableCell>
                  <TableCell>
                    {JSON.stringify(trial.proposal, null, 2)}
                  </TableCell>
                </TableRow>
              )}
              <TableRow>
                <TableCell>Started</TableCell>
                <TableCell>
                  {moment(trial.datetime_started).format("llll")}
                </TableCell>
              </TableRow>
              {trial.datetime_stopped && (
                <React.Fragment>
                  <TableRow>
                    <TableCell>Stopped</TableCell>
                    <TableCell>
                      {moment(trial.datetime_stopped).format("llll")}
                    </TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Duration</TableCell>
                    <TableCell>
                      {moment
                        .duration(
                          trial.datetime_stopped - trial.datetime_started
                        )
                        .humanize()}
                    </TableCell>
                  </TableRow>
                </React.Fragment>
              )}
            </TableBody>
          </Table>
        </Paper>
        <Button
          variant="contained"
          color="secondary"
          className={classes.mainBtn}
          onClick={() => this.props.history.goBack()}
        >
          Go Back
        </Button>
      </React.Fragment>
    )
  }

  renderLogsPlots() {
    const { logs } = this.state
    const { classes } = this.props

    return (
      // Show plots section if there are plots
      Object.values(logs.plots).length > 0 && (
        <React.Fragment>
          <Typography gutterBottom variant="h3">
            Plots
          </Typography>
          {Object.values(logs.plots).map((x, i) => {
            return (
              <Paper
                key={x.title}
                id={`plot-${i}`}
                className={classes.plotPaper}
              ></Paper>
            )
          })}
        </React.Fragment>
      )
    )
  }

  renderLogsMessages() {
    const { logs } = this.state
    const { classes } = this.props

    return (
      // Show messages section if there are messages
      Object.values(logs.messages).length > 0 && (
        <React.Fragment>
          <Typography gutterBottom variant="h3">
            Messages
          </Typography>
          <Paper className={classes.messagesPaper}>
            <List>
              {Object.values(logs.messages).map((x, i) => {
                return (
                  <ListItem key={(x.time || "") + x.message}>
                    <ListItemText
                      primary={x.message}
                      secondary={x.time ? x.time.toTimeString() : null}
                    />
                  </ListItem>
                )
              })}
            </List>
          </Paper>
        </React.Fragment>
      )
    )
  }

  render() {
    const { classes } = this.props
    const { trialId } = this.props.match.params
    const { logs, trial } = this.state

    return (
      <React.Fragment>
        <Typography gutterBottom variant="h2">
          Trial
          <span className={classes.headerSub}>{`(ID: ${trialId})`}</span>
        </Typography>
        {trial && this.renderDetails()}
        {logs &&
          (Object.values(logs.plots).length > 0 ||
            Object.values(logs.messages).length > 0) && (
            <Divider className={classes.divider} />
          )}
        {logs && logs.plots && this.renderLogsPlots()}
        {logs &&
          Object.values(logs.plots).length > 0 &&
          Object.values(logs.messages).length > 0 && (
            <Divider className={classes.divider} />
          )}
        {logs && logs.messages && this.renderLogsMessages()}
        {!(trial && logs) &&
          <LinearProgress color="secondary" />
        }
      </React.Fragment>
    )
  }
}

const mapDispatchToProps = {
  handleHeaderTitleChange: ConsoleActions.handleHeaderTitleChange,
  resetLoadingBar: ConsoleActions.resetLoadingBar,
}


export default compose(
  connect(null, mapDispatchToProps),
  withRouter,
  withStyles(styles),
)(TrialDetails)