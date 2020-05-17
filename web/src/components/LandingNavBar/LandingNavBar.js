import React, { Fragment } from "react"
import { Link, withRouter } from "react-router-dom"
import { compose } from "redux"
import { connect } from "react-redux"

import { withStyles } from "@material-ui/core/styles"
import Typography from "@material-ui/core/Typography"
import AppBar from "../LandingComponents/AppBar"
import Grid from "@material-ui/core/Grid"
import Hidden from "@material-ui/core/Hidden"
import MenuIcon from "@material-ui/icons/Menu"
import LandingNavigator from "./LandingNavigator"

// for login menu
import Button from "@material-ui/core/Button"
import IconButton from "@material-ui/core/IconButton"
import AvatarRegion from "components/RootComponents/AvatarRegion"

import Toolbar, { styles as toolbarStyles } from "../LandingComponents/Toolbar"
import Logo from "../../assets/LOGO_Rafiki-4.svg"

const styles = theme => ({
  LandingAppBar: {
    // borderBottom: `1px solid ${theme.palette.border}`,
    // backgroundColor: theme.palette.common.white,
    zIndex: theme.zIndex.drawer + 1,
  },
  title: {
    font: "500 25px Roboto,sans-serif",
    cursor: "pointer",
    color: "#FFF",
    textDecoration: "none",
    marginRight: 20,
  },
  placeholder: toolbarStyles(theme).root,
  toolbar: {
    justifyContent: "space-between",
  },
  left: {
    flex: 1,
    display: "flex",
    justifyContent: "flex-start",
    alignItems: "center",
  },
  logo: {
    height: 36,
    marginRight: 10,
  },
  leftLinkActive: {
    color: theme.palette.secondary.dark,
  },
  right: {
    flex: 1,
    display: "flex",
    justifyContent: "flex-end",
  },
  rightLink: {
    font: "300 18px Roboto,sans-serif",
    color: theme.palette.common.white,
    marginLeft: theme.spacing(5),
    textDecoration: "none",
    "&:hover": {
      color: theme.palette.secondary.main,
    },
  },
  rightLinkActive: {
    font: "300 18px Roboto,sans-serif",
    color: theme.palette.secondary.main,
    marginLeft: theme.spacing(5),
    textDecoration: "none",
  },
  linkSecondary: {
    color: theme.palette.secondary.main,
  },
  menuButton: {
    marginLeft: -theme.spacing(1),
    marginRight: theme.spacing(2),
  },
})

class LandingNavBar extends React.Component {
  state = {
    RootMobileOpen: false,
  }

  handleDrawerToggle = () => {
    // must use prevState
    this.setState(prevState => ({
      RootMobileOpen: !prevState.RootMobileOpen,
    }))
  }

  handleLogout = () => {
    localStorage.removeItem("token")
    localStorage.removeItem("expirationDate")
    this.props.history.push(`/`)
    window.location.reload()
  }

  render() {
    const { isAuthenticated, classes, location } = this.props

    const links = isAuthenticated ? (
      <Fragment>
        <Typography variant="h6">
          <Link
            to="/console/datasets/list-datasets"
            className={classes.rightLink}
          >
            {"Go To Console"}
          </Link>
        </Typography>
        <AvatarRegion
          isAuthenticated={isAuthenticated}
          logOut={this.handleLogout}
        />
      </Fragment>
    ) : (
      <Fragment>
        <Button
          color="inherit"
          style={{
            textDecoration: "none",
            fontSize: 16,
          }}
          component={Link}
          to={"/sign-in"}
        >
          {"Sign in"}
        </Button>
      </Fragment>
    )

    const navLinks = [
      /*{
        url: "/publications",
        label: "Publicationcs",
      },*/
      {
        url: "/contact",
        label: "Contact",
      },
      {
        // TODO change the docs link?
        url: "https://nginyc.github.io/rafiki/docs/latest/src/user/index.html",
        label: "Docs",
      },
    ]

    return (
      <div>
        <LandingNavigator
          PaperProps={{ style: { width: 250, backgroundColor: "rgb(0,0,0)" } }}
          variant="temporary"
          open={this.state.RootMobileOpen}
          onClose={this.handleDrawerToggle}
        />
        <AppBar position="fixed" className={classes.LandingAppBar}>
          <Toolbar className={classes.toolbar}>
            <Hidden mdUp>
              <Grid item>
                <IconButton
                  color="inherit"
                  aria-label="Open drawer"
                  onClick={this.handleDrawerToggle}
                  className={classes.menuButton}
                >
                  <MenuIcon />
                </IconButton>
              </Grid>
            </Hidden>
            <div className={classes.left}>
              <Link to="/">
                <img alt="logo" src={Logo} className={classes.logo} />
              </Link>
              <Link to="/" className={classes.title}>
                {"Panda"}
              </Link>
              <Hidden smDown>
                {navLinks.map((link, index) =>
                  /^https?:\/\//.test(link.url) ? ( // test if the url is external
                    <a
                      key={index}
                      href={link.url}
                      className={
                        location.pathname === link.url
                          ? classes.rightLinkActive
                          : classes.rightLink
                      }
                    >
                      {link.label}
                    </a>
                  ) : (
                    <Link
                      key={index}
                      to={link.url}
                      className={
                        location.pathname === link.url
                          ? classes.rightLinkActive
                          : classes.rightLink
                      }
                    >
                      {link.label}
                    </Link>
                  )
                )}
              </Hidden>
            </div>
            {links}
          </Toolbar>
        </AppBar>
        <div className={classes.placeholder} />
      </div>
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
)(LandingNavBar)
