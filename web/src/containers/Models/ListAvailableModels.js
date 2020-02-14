import React from "react"
import PropTypes from "prop-types"
import { connect } from "react-redux"
import { compose } from "redux"

import * as ConsoleActions from "../ConsoleAppFrame/actions"
import * as actions from "./actions"

import { withStyles } from "@material-ui/core/styles"
import Typography from "@material-ui/core/Typography"

import MainContent from "components/ConsoleContents/MainContent"
import ContentBar from "components/ConsoleContents/ContentBar"

import AvailableModelsTable from "components/ConsoleContents/MUITable"


const styles = theme => ({
  contentWrapper: {
    margin: "40px 16px",
    //position: "relative",
    minHeight: 200,
  },
})

class ListAvailableModels extends React.Component {
  /* use models/available first, 
    then ask @Xiong Yiyuan if he is ready with models/recommended,
    you will switch to models/recommended eventually
  */
  static propTypes = {
    classes: PropTypes.object.isRequired,
    handleHeaderTitleChange: PropTypes.func,
    AvailableModels: PropTypes.array,
    requestAvailableModelList: PropTypes.func,
    resetLoadingBar: PropTypes.func,
  }

  reloadListAvailModels = () => {
    this.props.requestAvailableModelList()
  }

  componentDidMount() {
    this.props.handleHeaderTitleChange("Model > List Available Models")
    this.props.requestAvailableModelList()
  }

  componentDidUpdate(prevProps, prevState) {}

  componentWillUnmount() {
    this.props.resetLoadingBar()
  }

  render() {
    const { classes, AvailableModels } = this.props

    const headCells = [
      { id: 'ModelID', numeric: false, disablePadding: true, label: 'Model-ID' },
      { id: 'Name', numeric: false, disablePadding: false, label: 'Name' },
      { id: 'Task', numeric: false, disablePadding: false, label: 'Task' },
      { id: 'Dependencies', numeric: true, disablePadding: false, label: 'Dependencies' },
      { id: 'UploadedAt', numeric: false, disablePadding: false, label: 'Uploaded At' },
      { id: 'ViewMore', numeric: false, disablePadding: true, label: 'View More'}
    ]

    return (
      <MainContent>
        <ContentBar
          needToList={true}
          barTitle="List Available Models"
          mainBtnText="Add Model"
          mainBtnLink="/console/models/upload-model"
          refreshAction={this.reloadListAvailModels}
        />
        <div className={classes.contentWrapper}>
          <Typography color="textSecondary" align="center">
            {AvailableModels.length === 0
              ? "You do not have any available models"
              : "Available Models"}
          </Typography>
          <AvailableModelsTable
            headCells={headCells}
            rows={AvailableModels}
            mode={"ListModels"}
          />
        </div>
      </MainContent>
    )
  }
}

const mapStateToProps = state => ({
  AvailableModels: state.ModelsReducer.AvailableModelList,
})

const mapDispatchToProps = {
  handleHeaderTitleChange: ConsoleActions.handleHeaderTitleChange,
  requestAvailableModelList: actions.requestAvailableModelList,
  resetLoadingBar: ConsoleActions.resetLoadingBar,
}

export default compose(
  connect(mapStateToProps, mapDispatchToProps),
  withStyles(styles)
)(ListAvailableModels)
