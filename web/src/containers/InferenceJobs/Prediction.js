import React from "react"
import PropTypes from "prop-types"
import { connect } from "react-redux"
import { compose } from "redux"
import { push } from "connected-react-router"
import PageviewIcon from "@material-ui/icons/Pageview"

import * as ConsoleActions from "../ConsoleAppFrame/actions"
import * as actions from "./actions"

import { withStyles } from "@material-ui/core/styles"
import Table from '@material-ui/core/Table';
import TableBody from '@material-ui/core/TableBody';
import TableCell from '@material-ui/core/TableCell';
import TableRow from '@material-ui/core/TableRow';
import TableHead from '@material-ui/core/TableHead';
import Typography from "@material-ui/core/Typography"
import IconButton from "@material-ui/core/IconButton"

import MainContent from "components/ConsoleContents/MainContent"
import ContentBar from "components/ConsoleContents/ContentBar"

// read query-string
import queryString from 'query-string'

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

class RunPrediction extends React.Component {
  static propTypes = {
    classes: PropTypes.object.isRequired,
    handleHeaderTitleChange: PropTypes.func,
    resetLoadingBar: PropTypes.func,
  }

  state = {
    app:"",
    appVersion:"",
    predictorHost:"",
    FormIsValid: false
  }

  componentDidMount() {
    this.props.handleHeaderTitleChange("Inference Jobs > Run Prediction")
    // read the query string from URL
    const values = queryString.parse(this.props.location.search)
    console.log("queryString parse: ", values)
    if (values.app && values.appVersion && values.predictorHost) {
      this.setState({
        app: values.app,
        appVersion: values.appVersion,
        predictorHost: values.predictorHost,
      })
    }
  }

  componentDidUpdate(prevProps, prevState) {}

  componentWillUnmount() {
    this.props.resetLoadingBar()
  }

  render() {
    const { classes } = this.props

    return (
      <React.Fragment>
        <MainContent>
          <ContentBar
            needToList={false}
            barTitle="Run Prediction"
          />
          <div className={classes.contentWrapper}>
            lala
          </div>
        </MainContent>
      </React.Fragment>
    )
  }
}

// const mapStateToProps = state => ({
//   InferenceJobsList: state.InferenceJobsReducer.InferenceJobsList,
// })

const mapDispatchToProps = {
  handleHeaderTitleChange: ConsoleActions.handleHeaderTitleChange,
  resetLoadingBar: ConsoleActions.resetLoadingBar,
  push: push,
}

export default compose(
  connect(null, mapDispatchToProps),
  withStyles(styles)
)(RunPrediction)
