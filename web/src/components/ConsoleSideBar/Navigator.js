import React from "react"
import PropTypes from "prop-types"
import classNames from "classnames"
import { compose } from "redux"
import { Link, withRouter } from "react-router-dom"

import { withStyles } from "@material-ui/core/styles"
import Divider from "@material-ui/core/Divider"
import Drawer from "@material-ui/core/Drawer"
import List from "@material-ui/core/List"
import ListItem from "@material-ui/core/ListItem"
import ListItemIcon from "@material-ui/core/ListItemIcon"
import ListItemText from "@material-ui/core/ListItemText"

// Icons
import CloudUpload from "@material-ui/icons/CloudUploadOutlined"
// dataset
import ListDsIcon from "@material-ui/icons/PhotoLibrary"
// model
import ListModelsIcon from "@material-ui/icons/LocalLibrary"
// train jobs
import ListTrainJobsIcon from "@material-ui/icons/Timeline"
import CreateTrainJobIcon from '@material-ui/icons/PostAdd';
// inference jobs
import DnsRoundedIcon from "@material-ui/icons/DnsRounded"
// import TrackChangesIcon from '@material-ui/icons/TrackChanges';
// for nested list
import Collapse from "@material-ui/core/Collapse"
import ExpandLess from "@material-ui/icons/ExpandLess"
import ExpandMore from "@material-ui/icons/ExpandMore"

// import Logo from "assets/Logo-Rafiki-cleaned.png"
import Logo from "assets/LOGO_Rafiki-4.svg"

// Navigator basic color dark blue specified in
// ConsoleTheme MuiDrawer's paper
const styles = theme => ({
  categoryHeader: {
    paddingTop: theme.spacing(2),
    paddingBottom: theme.spacing(2),
  },
  categoryHeaderPrimary: {
    color: theme.palette.common.white,
  },
  item: {
    paddingTop: 1,
    paddingBottom: 1,
    color: "rgba(255, 255, 255, 0.7)",
    "&:hover,&:focus": {
      backgroundColor: "rgba(255, 255, 255, 0.08)",
    },
  },
  itemCategory: {
    backgroundColor: "#232f3e",
    boxShadow: "0 -1px 0 #404854 inset",
    paddingTop: theme.spacing(2),
    paddingBottom: theme.spacing(2),
  },
  firebase: {
    fontSize: 24,
    color: theme.palette.common.white,
    backgroundColor: theme.palette.primary.main,
    fontFamily: theme.typography.fontFamily,
  },
  logo: {
    height: 28,
    marginRight: 10,
  },
  itemActiveItem: {
    color: theme.palette.secondary.main,
  },
  itemPrimary: {
    fontSize: "inherit",
  },
  itemIcon: {
    minWidth: "auto",
    marginRight: theme.spacing(2),
  },
  divider: {
    marginTop: theme.spacing(2),
  },
})

class Navigator extends React.Component {
  static propTypes = {
    classes: PropTypes.object.isRequired,
    location: PropTypes.object.isRequired,
    open: PropTypes.bool,
    onClose: PropTypes.func,
  }

  state = {
    DatasetsTableOpen: true,
    ModelsTableOpen: true,
    JobsTableOpen: true,
    InferenceJobsTableOpen: true,
  }

  handleClick = categoryHeader => {
    // case is collapseID
    switch (categoryHeader) {
      // CATxxx stands for Category xxx
      case "CATdatasets":
        this.setState(state => ({
          DatasetsTableOpen: !state.DatasetsTableOpen,
        }))
        break
      case "CATmodels":
        this.setState(state => ({ ModelsTableOpen: !state.ModelsTableOpen }))
        break
      case "CATtrainjobs":
        this.setState(state => ({ JobsTableOpen: !state.JobsTableOpen }))
        break
      case "CATinferencejobs":
        this.setState(state => ({
          InferenceJobsTableOpen: !state.InferenceJobsTableOpen,
        }))
        break
      default:
        this.setState(state => ({ JobsTableOpen: !state.DatasetsTableOpen }))
        return
    }
  }

  render() {
    const categories = [
      {
        id: "Datasets",
        collapseID: "CATdatasets",
        collapseIn: this.state.DatasetsTableOpen,
        children: [
          {
            id: "List Datasets",
            icon: <ListDsIcon />,
            pathname: "/console/datasets/list-datasets",
          },
          {
            id: "Upload Dataset",
            icon: <CloudUpload />,
            pathname: "/console/datasets/upload-dataset",
          },
          // {
          //   id: 'Delete Dataset',
          //   icon: <DeleteDsIcon />,
          //   pathname: "/console/datasets/delete-dataset"
          // },
        ],
      },
      {
        id: "Models",
        collapseID: "CATmodels",
        collapseIn: this.state.ModelsTableOpen,
        children: [
          {
            id: "List Models",
            icon: <ListModelsIcon />,
            pathname: "/console/models/list-models",
          },
          {
            id: "Upload Model",
            icon: <CloudUpload />,
            pathname: "/console/models/upload-model",
          },
        ],
      },
      {
        id: "Train Jobs",
        collapseID: "CATtrainjobs",
        collapseIn: this.state.JobsTableOpen,
        children: [
          {
            id: "List TrainJobs",
            icon: <ListTrainJobsIcon />,
            pathname: "/console/jobs/list-train-jobs",
          },
          {
            id: "Create Train Job",
            icon: <CreateTrainJobIcon />,
            pathname: "/console/jobs/create-train-job",
          },
        ],
      },
      {
        id: "Inference Jobs",
        collapseID: "CATinferencejobs",
        collapseIn: this.state.InferenceJobsTableOpen,
        children: [
          {
            id: "List Inference Jobs",
            icon: <DnsRoundedIcon />,
            pathname: "/console/inferencejobs/list-inferencejobs",
          },
          // {
          //   id: "Run Prediction",
          //   icon: <TrackChangesIcon />,
          //   pathname: "/console/inferencejobs/run-prediction",
          // },
        ],
      },
    ]

    const {
      classes,
      location,
      staticContext,
      open,
      onClose,
      ...other
    } = this.props

    return (
      <Drawer variant="permanent" open={open} onClose={onClose} {...other}>
        <List disablePadding>
          <ListItem
            component={Link}
            to="/"
            className={classNames(
              classes.firebase,
              classes.item,
              classes.itemCategory
            )}
          >
            <img alt="logo" src={Logo} className={classes.logo} />
            PANDA
          </ListItem>

          {categories.map(({ id, collapseID, collapseIn, children }) => (
            <React.Fragment key={id}>
              <ListItem
                button
                onClick={() => this.handleClick(collapseID)}
                className={classes.categoryHeader}
              >
                <ListItemText
                  classes={{
                    primary: classes.categoryHeaderPrimary,
                  }}
                >
                  {id}
                </ListItemText>
                {collapseIn ? (
                  <ExpandLess
                    style={{
                      color: "white",
                    }}
                  />
                ) : (
                  <ExpandMore
                    style={{
                      color: "white",
                    }}
                  />
                )}
              </ListItem>
              <Collapse in={collapseIn} timeout="auto" unmountOnExit>
                {children.map(({ id: childId, icon, pathname }) => (
                  <ListItem
                    key={childId}
                    button
                    onClick={onClose}
                    component={Link}
                    to={pathname}
                    className={classNames(
                      classes.item,
                      location.pathname === pathname && classes.itemActiveItem
                    )}
                  >
                    <ListItemIcon className={classes.itemIcon}>
                      {icon}
                    </ListItemIcon>
                    <ListItemText
                      classes={{
                        primary: classes.itemPrimary,
                      }}
                    >
                      {childId}
                    </ListItemText>
                  </ListItem>
                ))}
              </Collapse>
              <Divider className={classes.divider} />
            </React.Fragment>
          ))}
        </List>
      </Drawer>
    )
  }
}

export default compose(withRouter, withStyles(styles))(Navigator)
