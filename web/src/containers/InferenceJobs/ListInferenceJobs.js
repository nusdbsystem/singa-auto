import React from "react"
import PropTypes from "prop-types"
import { connect } from "react-redux"
import { compose } from "redux"
import { push } from "connected-react-router"
import PageviewIcon from "@material-ui/icons/Pageview"

import * as ConsoleActions from "../ConsoleAppFrame/actions"
import * as actions from "./actions"

import { withStyles } from "@material-ui/core/styles"
import {
  Table,
  Typography,
  IconButton,
  TableHead,
  TableBody,
  TableRow,
  TableCell,
} from "@material-ui/core"

// Third parts
import * as moment from "moment"

import MainContent from "components/ConsoleContents/MainContent"
import ContentBar from "components/ConsoleContents/ContentBar"

const styles = theme => ({
  block: {
    display: "block",
  },
  addDS: {
    marginRight: theme.spacing(1),
  },
  contentWrapper: {
    margin: "40px 16px",
    //position: "relative",
    minHeight: 200,
  },
})

class ListInferenceJobs extends React.Component {
  static propTypes = {
    classes: PropTypes.object.isRequired,
    handleHeaderTitleChange: PropTypes.func,
    resetLoadingBar: PropTypes.func,
  }

  componentDidMount() {
    this.props.handleHeaderTitleChange("Inference Jobs > List Inference Jobs")
    this.props.getInferenceJobsList()
  }

  componentDidUpdate(prevProps, prevState) {}

  componentWillUnmount() {
    this.props.resetLoadingBar()
  }

  render() {
    const { classes, InferenceJobsList } = this.props

    return (
      <React.Fragment>
        <MainContent>
          <ContentBar
            needToList={false}
            barTitle="List Inference Jobs"
          />
          <div className={classes.contentWrapper}>
            <Typography color="textSecondary" align="center">
              {InferenceJobsList.length === 0
                ? "You do not have any inference jobs for this user"
                : "Inference Jobs"}
            </Typography>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>View</TableCell>
                  <TableCell>Inference Job ID</TableCell>
                  <TableCell>App Name</TableCell>
                  <TableCell>App Version</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Started</TableCell>
                  <TableCell>Stopped</TableCell>
                  <TableCell>Train Job ID</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {InferenceJobsList.map(x => {
                  return (
                    <TableRow key={x.id} hover>
                      <TableCell padding="default">
                        <IconButton
                          onClick={() => {
                            const link = "/console/inferencejobs/running_job/:app/:appVersion"
                              .replace(":app", x.app)
                              .replace(":appVersion", x.app_version)
                            this.props.push(link)
                          }}
                        >
                          <PageviewIcon />
                        </IconButton>
                      </TableCell>
                      <TableCell>{x.id.slice(0, 8)}</TableCell>
                      <TableCell>{x.app}</TableCell>
                      <TableCell>{x.app_version}</TableCell>
                      <TableCell>{x.status}</TableCell>
                      <TableCell>
                        {x.datetime_started
                          ? moment(x.datetime_started).fromNow()
                          : "-"}
                      </TableCell>
                      <TableCell>
                        {x.datetime_stopped
                          ? moment(x.datetime_stopped).fromNow()
                          : "-"}
                      </TableCell>
                      <TableCell>{x.train_job_id.slice(0, 8)}</TableCell>
                    </TableRow>
                  )
                })}
              </TableBody>
            </Table>
          </div>
        </MainContent>
      </React.Fragment>
    )
  }
}

const mapStateToProps = state => ({
  InferenceJobsList: state.InferenceJobsReducer.InferenceJobsList,
})

const mapDispatchToProps = {
  handleHeaderTitleChange: ConsoleActions.handleHeaderTitleChange,
  resetLoadingBar: ConsoleActions.resetLoadingBar,
  getInferenceJobsList: actions.fetchGetInferencejob,
  push: push,
}

export default compose(
  connect(mapStateToProps, mapDispatchToProps),
  withStyles(styles)
)(ListInferenceJobs)
