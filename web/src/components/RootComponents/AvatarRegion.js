import React from "react"
import PropTypes from "prop-types"

import { makeStyles } from '@material-ui/core/styles';
import IconButton from "@material-ui/core/IconButton";
import Menu from "@material-ui/core/Menu";
import Avatar from '@material-ui/core/Avatar';
import AccountIcon from '@material-ui/icons/Person';
import AppBarMenuItems from "components/LandingNavBar/AppBarMenuItems"

const useStyles = makeStyles(theme => {
  // console.log(theme)
  return {
    avatar: {
      margin: 10,
      color: '#fff',
      backgroundColor: theme.palette.secondary.main,
    },
    iconButtonAvatar: {
      padding: 4,
      marginLeft: theme.spacing(1),
      textDecoration: "none"
    },
  }
})


const AvatarRegion = props => {
  const classes = useStyles()
  const {
    isAuthenticated,
    anchorElId,
    openMenu,
    closeMenu,
    logOut
  } = props

  return (
    <>
    <IconButton
      aria-haspopup="true"
      aria-label="More"
      aria-owns="Open right Menu"
      color="inherit"
      id="loginMenuButton"
      onClick={openMenu}
      className={classes.iconButtonAvatar}
    >
      <Avatar
        className={classes.avatar}
        style={{
          backgroundColor: "orange" //bgColor
        }}
      >
        <AccountIcon />
      </Avatar>
    </IconButton>
    <Menu
      anchorEl={
        (anchorElId && document.getElementById(anchorElId)) ||
        document.body
      }
      id="menuRight"
      onClose={closeMenu}
      open={!!anchorElId}
    >
      <AppBarMenuItems
        isAuth={isAuthenticated}
        logout={logOut}
        onClick={closeMenu}
      />
    </Menu>
    </>
  )
}

AvatarRegion.propTypes = {
  isAuthenticated: PropTypes.bool.isRequired,
  anchorElId: PropTypes.oneOfType([
    PropTypes.string,
    PropTypes.bool,
  ]).isRequired,
  openMenu: PropTypes.func.isRequired,
  closeMenu: PropTypes.func.isRequired,
  logOut: PropTypes.func.isRequired,
}

export default AvatarRegion
