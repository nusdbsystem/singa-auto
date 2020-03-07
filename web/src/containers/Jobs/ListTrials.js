import React from "react"
import PropTypes from "prop-types"

import { connect } from "react-redux"
import { compose } from "redux"
import { push } from "connected-react-router"

// Material UI
import { withStyles } from "@material-ui/core/styles"
import Table from '@material-ui/core/Table';
import TableHead from '@material-ui/core/TableHead';
import TableBody from '@material-ui/core/TableBody';
import TableRow from '@material-ui/core/TableRow';
import TableCell from '@material-ui/core/TableCell';
import TableContainer from '@material-ui/core/TableContainer';

import IconButton from '@material-ui/core/IconButton';
import Typography from '@material-ui/core/Typography';

import PageviewIcon from "@material-ui/icons/Pageview"

import * as ConsoleActions from "../ConsoleAppFrame/actions"
import * as jobsActions from "./actions"
import * as inferenceJobActions from "../InferenceJobs/actions"

// Import Layout
import MainContent from "components/ConsoleContents/MainContent"
import ContentBar from "components/ConsoleContents/ContentBar"

// Third parts
import * as moment from "moment"

/* ListJobs are able to view trials and Trial details*/

const styles = theme => ({
  contentWrapper: {
    margin: "40px 16px",
    //position: "relative",
    minHeight: 200,
  },
  table: {
    minWidth: 750,
  },
})

class ListTrials extends React.Component {
  static propTypes = {
    classes: PropTypes.object.isRequired,
    handleHeaderTitleChange: PropTypes.func,
    jobsList: PropTypes.array,
    trialsList: PropTypes.array,
    requestTrialsListOfJob: PropTypes.func,
    resetLoadingBar: PropTypes.func,
  }

  reloadListOfTrials = () => {
    console.log("reload trials...")
    const { app, appVersion } = this.props.match.params
    this.props.requestTrialsListOfJob(app, appVersion)
  }

  componentDidMount() {
    const { app, appVersion } = this.props.match.params
    this.props.handleHeaderTitleChange("Training Jobs > Jobs List > List Trials")
    this.props.requestTrialsListOfJob(app, appVersion)
  }

  componentDidUpdate(prevProps, prevState) {}

  componentWillUnmount() {
    this.props.resetLoadingBar()
  }

  render() {
    const { classes, jobsList, match } = this.props

    const { appId, app, appVersion } = match.params
    // eslint-disable-next-line
    const job = jobsList.find(job => job.id == appId) // id & appId might not be same type
    let trialsList = []
    // eslint-disable-next-line
    if (job != undefined) {
      trialsList = job.trials || []
    }
    return (
      <React.Fragment>
        <MainContent>
          <ContentBar
            needToList={true}
            barTitle="Selected Train Job"
            mainBtnText="Create Inference Job"
            mainBtnLink={`/console/inferencejobs/${appId}/${app}/${appVersion}/create_inference_job`}
            refreshAction={this.reloadListOfTrials}
          />
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell> Train Job ID </TableCell>
                  <TableCell> App Name </TableCell>
                  <TableCell> App Version</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell> {appId}</TableCell>
                  <TableCell> {app} </TableCell>
                  <TableCell> {appVersion}</TableCell>
                </TableRow>
              </TableBody>
            </Table>
            <div className={classes.contentWrapper}>
              <Typography color="textSecondary" align="center">
                {trialsList.length === 0
                  ? "You do not have any trials for this train job"
                  : "Trials for this train job"}
              </Typography>
              <TableContainer>
                <Table
                  className={classes.table}
                  aria-labelledby="tableTitle"
                  size={'medium'}
                  aria-label="enhanced table"
                >
                <TableHead>
                  <TableRow>
                    <TableCell>Details</TableCell>
                    <TableCell>Model Name</TableCell>
                    <TableCell>Trial No</TableCell>
                    <TableCell>Score</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Started</TableCell>
                    <TableCell>Stopped</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {trialsList.map(x => {
                    return (
                      <TableRow key={x.id} hover>
                        <TableCell padding="default">
                          <IconButton
                            onClick={() => {
                              // click to see individual trial
                              this.props.push(
                                `/console/jobs/trials/${x.id}`
                              )
                            }}
                          >
                            <PageviewIcon />
                          </IconButton>
                        </TableCell>
                        <TableCell>{x.model_name}</TableCell>
                        <TableCell>{x.no}</TableCell>
                        <TableCell>
                          {x.score !== null ? parseFloat(x.score).toFixed(3) : "-"}
                        </TableCell>
                        <TableCell>{x.status}</TableCell>
                        <TableCell>
                          {moment(x.datetime_started).fromNow()}
                        </TableCell>
                        <TableCell>
                          {x.datetime_stopped
                            ? moment(x.datetime_stopped).fromNow()
                            : "null"}
                        </TableCell>
                      </TableRow>
                    )
                  })}
                </TableBody>
              </Table>
            </TableContainer>
          </div>
        </MainContent>
      </React.Fragment>
    )
  }
}

const mapStateToProps = state => ({
  jobsList: state.JobsReducer.jobsList,
})

const mapDispatchToProps = {
  createInferenceJob: inferenceJobActions.postCreateInferenceJob,
  handleHeaderTitleChange: ConsoleActions.handleHeaderTitleChange,
  requestTrialsListOfJob: jobsActions.requestTrialsListOfJob,
  resetLoadingBar: ConsoleActions.resetLoadingBar,
  push: push,
}

export default compose(
  connect(mapStateToProps, mapDispatchToProps),
  withStyles(styles)
)(ListTrials)
