import React from "react"
import PropTypes from "prop-types"
import { Link } from "react-router-dom"

import { makeStyles } from "@material-ui/core/styles"
import IconButton from "@material-ui/core/IconButton"
import Menu from "@material-ui/core/Menu"
import Avatar from "@material-ui/core/Avatar"
import AccountIcon from "@material-ui/icons/Person"
import MenuItem from "@material-ui/core/MenuItem"

const useStyles = makeStyles(theme => {
  // console.log(theme)
  return {
    avatar: {
      margin: 10,
      color: "#fff",
      backgroundColor: theme.palette.secondary.main,
    },
    iconButtonAvatar: {
      padding: 4,
      marginLeft: theme.spacing(1),
      textDecoration: "none",
    },
    link: {
      textDecoration: "none",
      color: "inherit",
    },
  }
})

const AvatarRegion = props => {
  const classes = useStyles()

  const [anchorEl, setAnchorEl] = React.useState(null)

  const openMenu = event => {
    setAnchorEl(event.currentTarget)
  }

  const closeMenu = () => {
    setAnchorEl(null)
  }

  const { isAuthenticated, logOut } = props

  return (
    <React.Fragment>
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
            backgroundColor: "orange", //bgColor
          }}
        >
          <AccountIcon />
        </Avatar>
      </IconButton>
      <Menu
        anchorEl={anchorEl}
        id="menuRight"
        keepMounted
        onClose={closeMenu}
        open={Boolean(anchorEl)}
      >
        <MenuItem
          className={classes.link}
          disabled
          component={Link}
          to={`#/profile/${isAuthenticated}`}
          onClick={() => {
            closeMenu()
          }}
        >
          My account
        </MenuItem>
        <MenuItem
          onClick={() => {
            closeMenu()
            logOut()
          }}
        >
          Logout
        </MenuItem>
      </Menu>
    </React.Fragment>
  )
}

AvatarRegion.propTypes = {
  isAuthenticated: PropTypes.bool.isRequired,
  logOut: PropTypes.func.isRequired,
}

export default AvatarRegion
