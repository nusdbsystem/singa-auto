import React from "react"
import PropTypes from "prop-types"
import { withRouter } from "react-router-dom"
import { connect } from "react-redux"
import { compose } from "redux"

import AppBar from "@material-ui/core/AppBar"
import Grid from "@material-ui/core/Grid"
import Hidden from "@material-ui/core/Hidden"
import MenuIcon from "@material-ui/icons/Menu"
import Toolbar from "@material-ui/core/Toolbar"
import Typography from "@material-ui/core/Typography"
import { withStyles } from "@material-ui/core/styles"

// for login menu
import IconButton from "@material-ui/core/IconButton"
import AvatarRegion from "components/RootComponents/AvatarRegion"

const lightColor = "rgba(255, 255, 255, 0.7)"

const styles = theme => ({
  menuButton: {
    marginLeft: -theme.spacing(1),
  },
  link: {
    textDecoration: "none",
    color: lightColor,
    "&:hover": {
      color: theme.palette.common.white,
    },
  },
})

class Header extends React.Component {
  static propTypes = {
    classes: PropTypes.object.isRequired,
    title: PropTypes.string.isRequired,
    onDrawerToggle: PropTypes.func.isRequired,
    isAuthenticated: PropTypes.bool.isRequired,
  }

  handleLogout = () => {
    // console.log("logging out, clearing token")
    localStorage.removeItem("token")
    localStorage.removeItem("expirationDate")
    this.props.history.push(`/`)
    window.location.reload()
  }

  render() {
    const {
      classes,
      title,
      onDrawerToggle,
      isAuthenticated,
      //initials,
      //bgColor
    } = this.props

    return (
      <AppBar color="primary" position="sticky">
        <Toolbar>
          <Grid container spacing={1} alignItems="center">
            <Hidden smUp>
              <Grid item>
                <IconButton
                  color="inherit"
                  aria-label="Open drawer"
                  onClick={onDrawerToggle}
                  className={classes.menuButton}
                >
                  <MenuIcon />
                </IconButton>
              </Grid>
            </Hidden>
            <Grid item xs>
              <Typography color="inherit" variant="h5" component="h1">
                {title}
              </Typography>
            </Grid>
            <Grid item>
              <Typography className={classes.link} component="a" href="#">
                Go to docs
              </Typography>
            </Grid>
            <Grid item>
              <AvatarRegion
                isAuthenticated={isAuthenticated}
                logOut={this.handleLogout}
              />
            </Grid>
          </Grid>
        </Toolbar>
      </AppBar>
    )
  }
}

const mapStateToProps = state => ({
  isAuthenticated: state.Root.token !== null,
  // initials: state.firebaseReducer.profile.initials,
  // bgColor: state.firebaseReducer.profile.color
})

export default compose(
  connect(mapStateToProps),
  withRouter,
  withStyles(styles)
)(Header)
