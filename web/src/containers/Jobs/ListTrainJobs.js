import React from "react"
import PropTypes from "prop-types"

import { connect } from "react-redux"
import { compose } from "redux"
import { push } from "connected-react-router"

import { withStyles } from "@material-ui/core/styles"
import PageviewIcon from "@material-ui/icons/Pageview"

import * as ConsoleActions from "../ConsoleAppFrame/actions"
import * as jobsActions from "./actions"

// Material UI
import Table from '@material-ui/core/Table';
import TableHead from '@material-ui/core/TableHead';
import TableBody from '@material-ui/core/TableBody';
import TableRow from '@material-ui/core/TableRow';
import TableCell from '@material-ui/core/TableCell';
import TableContainer from '@material-ui/core/TableContainer';

import IconButton from '@material-ui/core/IconButton';
import Typography from '@material-ui/core/Typography';

// Import Layout
import MainContent from "components/ConsoleContents/MainContent"
import ContentBar from "components/ConsoleContents/ContentBar"

// Third parts
import * as moment from "moment"

/* ListTrainJobs are able to view trials and Trial details*/

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

class ListTrainJobs extends React.Component {
  static propTypes = {
    classes: PropTypes.object.isRequired,
    handleHeaderTitleChange: PropTypes.func,
    JobsList: PropTypes.array,
    requestJobsList: PropTypes.func,
    resetLoadingBar: PropTypes.func,
  }

  reloadJobs = () => {
    this.props.requestJobsList()
  }

  componentDidMount() {
    this.props.handleHeaderTitleChange("Training Jobs > Jobs List")
    this.props.requestJobsList()
  }

  componentDidUpdate(prevProps, prevState) {}

  componentWillUnmount() {
    this.props.resetLoadingBar()
  }

  render() {
    const { classes, JobsList } = this.props

    return (
      <React.Fragment>
        <MainContent>
          <ContentBar
            needToList={true}
            barTitle="Train Jobs by user"
            mainBtnText="Create Train Job"
            mainBtnLink="/console/jobs/create-train-job"
            refreshAction={this.reloadJobs}
          />
          <div className={classes.contentWrapper}>
            <Typography color="textSecondary" align="center">
              {JobsList.length === 0
                ? "You do not have any train jobs"
                : "Train Jobs"}
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
                  <TableCell> View Trials </TableCell>
                  <TableCell> App Name </TableCell>
                  <TableCell> App Version</TableCell>
                  <TableCell> Task </TableCell>
                  <TableCell> Budget </TableCell>
                  <TableCell> Started</TableCell>
                  <TableCell> Stopped </TableCell>
                  <TableCell> Status </TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {JobsList.map(x => {
                  return (
                    <TableRow key={x.id} hover>
                      <TableCell padding="none">
                        <IconButton
                          onClick={() => {
                            const link = "/console/jobs/trials/:appId/:app/:appVersion"
                              .replace(":appId", x.id)
                              .replace(":app", x.app)
                              .replace(":appVersion", x.app_version)
                            this.props.push(link)
                          }}
                        >
                          <PageviewIcon />
                        </IconButton>
                      </TableCell>
                      <TableCell>{x.app}</TableCell>
                      <TableCell>{x.app_version}</TableCell>
                      <TableCell>{x.task}</TableCell>
                      <TableCell>{JSON.stringify(x.budget)}</TableCell>
                      <TableCell>
                        {moment(x.datetime_started).fromNow()}
                      </TableCell>
                      <TableCell>
                        {x.datetime_stopped
                          ? moment(x.datetime_stopped).fromNow()
                          : "-"}
                      </TableCell>
                      <TableCell>{x.status}</TableCell>
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
  JobsList: state.JobsReducer.jobsList,
})

const mapDispatchToProps = {
  handleHeaderTitleChange: ConsoleActions.handleHeaderTitleChange,
  requestJobsList: jobsActions.requestJobsList,
  resetLoadingBar: ConsoleActions.resetLoadingBar,
  push: push,
}

export default compose(
  connect(mapStateToProps, mapDispatchToProps),
  withStyles(styles)
)(ListTrainJobs)
