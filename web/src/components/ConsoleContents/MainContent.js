import React from "react"
import PropTypes from "prop-types"
import { compose } from "redux"
import { Link, withRouter } from "react-router-dom"

import Paper from "@material-ui/core/Paper"
import { withStyles } from "@material-ui/core/styles"

// zoom addicon
import Fab from "@material-ui/core/Fab"
import Zoom from "@material-ui/core/Zoom"
import ListDSIcon from "@material-ui/icons/FormatListBulleted"
import AddIcon from "@material-ui/icons/Add"

const styles = theme => ({
  paper: {
    maxWidth: 1300,
    margin: "auto",
    overflow: "hidden",
    marginBottom: 20,
    position: "relative",
    paddingBottom: 80,
  },
  fab: {
    position: "absolute",
    bottom: theme.spacing(3),
    right: theme.spacing(3),
    zIndex: 10,
  },
  extendedIcon: {
    marginRight: theme.spacing(1),
  },
})

class MainContent extends React.Component {
  static propTypes = {
    classes: PropTypes.object.isRequired,
    children: PropTypes.node,
    location: PropTypes.object,
  }

  render() {
    const { classes, children, location } = this.props

    const customizeZoomIcon = pathname => {
      switch (pathname) {
        case "/console/datasets/list-datasets":
          console.log("pathname is list-datasets: ", pathname)
          return (
            <Zoom in={true} unmountOnExit>
              <Fab
                className={classes.fab}
                color="primary"
                component={Link}
                to="/console/datasets/upload-dataset"
              >
                <AddIcon />
              </Fab>
            </Zoom>
          )
        case "/console/datasets/upload-dataset":
          return (
            <Zoom in={true} unmountOnExit>
              <Fab
                variant="extended"
                className={classes.fab}
                color="primary"
                component={Link}
                to="/console/datasets/list-datasets"
              >
                <ListDSIcon className={classes.extendedIcon} />
                List Datasets
              </Fab>
            </Zoom>
          )
        case "/console/models/list-models":
          console.log("pathname is about models: ", pathname)
          return (
            <Zoom in={true} unmountOnExit>
              <Fab
                className={classes.fab}
                color="primary"
                component={Link}
                to="/console/models/upload-model"
              >
                <AddIcon />
              </Fab>
            </Zoom>
          )
        case "/console/models/upload-model":
          return (
            <Zoom in={true} unmountOnExit>
              <Fab
                variant="extended"
                className={classes.fab}
                color="primary"
                component={Link}
                to="/console/models/list-models"
              >
                <ListDSIcon className={classes.extendedIcon} />
                List Models
                </Fab>
            </Zoom>
          )
        case "/console/jobs/list-train-jobs":
          return (
            <Zoom in={true} unmountOnExit>
              <Fab
                className={classes.fab}
                color="primary"
                component={Link}
                to="/console/jobs/create-train-job"
              >
                <AddIcon />
              </Fab>
            </Zoom>
          )
        case "/console/jobs/create-train-job":
          return (
            <Zoom in={true} unmountOnExit>
              <Fab
                variant="extended"
                className={classes.fab}
                color="primary"
                component={Link}
                to="/console/jobs/list-train-jobs"
              >
                <ListDSIcon className={classes.extendedIcon} />
                List Train Jobs
                </Fab>
            </Zoom>
          )
        default:
          return (
            <Zoom in={true} unmountOnExit>
              <Fab
                variant="extended"
                className={classes.fab}
                color="primary"
                component={Link}
                to="/console/datasets/list-datasets"
              >
                <ListDSIcon className={classes.extendedIcon} />
                List Datasets
              </Fab>
            </Zoom>
          )
      }
    }

    return (
      <Paper className={classes.paper}>
        {children}
        {customizeZoomIcon(location.pathname)}
      </Paper>
    )
  }
}

export default compose(withRouter, withStyles(styles))(MainContent)
