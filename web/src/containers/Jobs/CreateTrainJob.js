import React from "react"

import { withStyles } from "@material-ui/core/styles"
import { compose } from "redux"
import { connect } from "react-redux"

// Material UI
import { Toolbar, Typography, Grid } from "@material-ui/core"

import * as ModelActions from "../Models/actions"
import * as DatasetActions from "../Datasets/actions"
import * as ConsoleActions from "../ConsoleAppFrame/actions"
import * as actions from "./actions"

// Import Layout
import MainContent from "components/ConsoleContents/MainContent"
import ContentBar from "components/ConsoleContents/ContentBar"

import CreateTrainJobForm from "components/ConsoleForms/CreateTrainJobForm"

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

class CreateTrainJob extends React.Component {
  componentDidMount() {
    this.props.requestListDS()
    this.props.requestAvailableModelList()
  }

  render() {
    const { classes, DatasetsList, AvailableModelList } = this.props

    return (
      <React.Fragment>
        <MainContent>
          <ContentBar>
            <Toolbar>
              <Grid
                container
                spacing={10}
                justify="space-between"
                alignItems="center"
              >
                <Grid item>
                  <Typography variant="h5" gutterBottom>
                    Create Train Jobs
                  </Typography>
                </Grid>
              </Grid>
            </Toolbar>
          </ContentBar>
          <div className={classes.contentWrapper}>
            <CreateTrainJobForm
              datasets={DatasetsList}
              models={AvailableModelList}
              postCreateTrainJob={this.props.postCreateTrainJob}
            />
          </div>
        </MainContent>
      </React.Fragment>
    )
  }
}

const mapStateToProps = state => ({
  DatasetsList: state.DatasetsReducer.DatasetList,
  AvailableModelList: state.ModelsReducer.AvailableModelList,
})

const mapDispatchToProps = {
  handleHeaderTitleChange: ConsoleActions.handleHeaderTitleChange,
  postCreateTrainJob: actions.createTrainJob,
  requestListDS: DatasetActions.requestListDS,
  resetLoadingBar: ConsoleActions.resetLoadingBar,
  requestAvailableModelList: ModelActions.requestAvailableModelList,
}

export default compose(
  connect(mapStateToProps, mapDispatchToProps),
  withStyles(styles)
)(CreateTrainJob)
