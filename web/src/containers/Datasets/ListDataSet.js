import React from "react"
import PropTypes from "prop-types"
import { connect } from "react-redux"
import { compose } from "redux"

import * as ConsoleActions from "../ConsoleAppFrame/actions"
import * as actions from "./actions"

import { withStyles } from "@material-ui/core/styles"
import Typography from "@material-ui/core/Typography"

import MainContent from "components/Console/ConsoleContents/MainContent"
import ContentBar from "components/Console/ConsoleContents/ContentBar"

import ListDataSetTable from "components/Console/ConsoleContents/MUITable"


const styles = theme => ({
  contentWrapper: {
    margin: "40px 16px",
    //position: "relative",
    minHeight: 200,
  },
})

class ListDataSet extends React.Component {
  static propTypes = {
    classes: PropTypes.object.isRequired,
    handleHeaderTitleChange: PropTypes.func,
    DatasetList: PropTypes.array,
    requestListDS: PropTypes.func,
    resetLoadingBar: PropTypes.func,
  }

  reloadSizeAndDS = () => {
    this.props.requestListDS()
  }

  componentDidMount() {
    this.props.handleHeaderTitleChange("Dataset > List Dataset")
    this.props.requestListDS()
  }

  componentDidUpdate(prevProps, prevState) {}

  componentWillUnmount() {
    this.props.resetLoadingBar()
  }

  render() {
    const { classes, DatasetList } = this.props

    const headCells = [
      { id: 'ID', numeric: false, disablePadding: true, label: 'ID' },
      { id: 'Name', numeric: false, disablePadding: false, label: 'Name' },
      { id: 'Task', numeric: false, disablePadding: false, label: 'Task' },
      { id: 'Size', numeric: true, disablePadding: false, label: 'Size (bytes)' },
      { id: 'UploadedAt', numeric: false, disablePadding: false, label: 'Uploaded At' },
      { id: 'ViewMore', numeric: false, disablePadding: true, label: 'View More'}
    ]

    return (
      <MainContent>
        <ContentBar
          needToList={true}
          barTitle="List Dataset"
          mainBtnText="Add Dataset"
          mainBtnLink="/console/datasets/upload-datasets"
          refreshAction={this.reloadSizeAndDS}
        />
        <div className={classes.contentWrapper}>
          <Typography color="textSecondary" align="center">
            {DatasetList.length === 0
              ? "You do not have any dataset"
              : "Datasets"}
          </Typography>
          <ListDataSetTable
            headCells={headCells}
            rows={DatasetList}
          />
        </div>
      </MainContent>
    )
  }
}

const mapStateToProps = state => ({
  DatasetList: state.DatasetsReducer.DatasetList,
})

const mapDispatchToProps = {
  handleHeaderTitleChange: ConsoleActions.handleHeaderTitleChange,
  requestListDS: actions.requestListDS,
  resetLoadingBar: ConsoleActions.resetLoadingBar,
}

export default compose(
  connect(mapStateToProps, mapDispatchToProps),
  withStyles(styles)
)(ListDataSet)
