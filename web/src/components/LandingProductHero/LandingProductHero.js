import React from "react"
import PropTypes from "prop-types"
import { withStyles } from "@material-ui/core/styles"
import { Link } from "react-router-dom"
import Button from "../LandingComponents/Button"
import Typography from "../LandingComponents/Typography"
import ProductHeroLayout from "./LandingProductHeroLayout"
import heroImage from "../../assets/harli-marten-n7a2OJDSZns-unsplash.jpg"

const backgroundImage = heroImage

const styles = theme => ({
  background: {
    backgroundImage: `url(${heroImage})`,
    backgroundColor: "#333333", // Average color of the background image.
    backgroundPosition: "center",
  },
  button: {
    //minWidth: 200,
  },
  h5: {
    marginBottom: theme.spacing(4),
    marginTop: theme.spacing(4),
    [theme.breakpoints.up("sm")]: {
      marginTop: theme.spacing(10),
    },
  },
  more: {
    marginTop: theme.spacing(2),
  },
})

function ProductHero(props) {
  const { classes } = props

  return (
    <ProductHeroLayout backgroundClassName={classes.background}>
      {/* Increase the network loading priority of the background image. */}
      <img
        style={{ display: "none" }}
        src={backgroundImage}
        alt="increase priority"
      />
      <Typography color="inherit" align="center" variant="h2" marked="center">
        Panda
      </Typography>
      <Typography
        color="inherit"
        align="center"
        variant="h5"
        className={classes.h5}
      >
        Panda is a distributed system that trains machine learning (ML)
        models <br />
        and deploys trained models, built with ease-of-use in mind.
      </Typography>
      <Button
        color="secondary"
        variant="contained"
        size="large"
        className={classes.button}
        component={Link}
        to={`/console/datasets/list-datasets`}
      >
        Try Panda
      </Button>
      <Typography variant="body2" color="inherit" className={classes.more}>
        Discover the experience
      </Typography>
    </ProductHeroLayout>
  )
}

ProductHero.propTypes = {
  classes: PropTypes.object.isRequired,
}

export default withStyles(styles)(ProductHero)
