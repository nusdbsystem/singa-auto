import React from 'react';
import PropTypes from 'prop-types';

import { withStyles } from '@material-ui/core/styles';
import Typography from '@material-ui/core/Typography';
import Grid from '@material-ui/core/Grid';
import MenuItem from '@material-ui/core/MenuItem';
// The select prop makes the text field 
// use the Select component internally.
import TextField from '@material-ui/core/TextField';

const styles = theme => ({
  menu: {
    width: 250,
  },
  textField: {
    marginLeft: theme.spacing(1),
    marginRight: theme.spacing(1),
    width: 250,
  },
})


class ModelClassSelect extends React.Component {
  static propTypes = {
    classes: PropTypes.object.isRequired,
    title: PropTypes.string,
    modelClass: PropTypes.array,
    onHandleChange: PropTypes.func,
  }

  render() {
    const {
      classes,
      title,
      modelClass,
      onHandleChange,
    } = this.props;

    return (
      <React.Fragment>
        <Typography variant="h5" gutterBottom align="center">
          {title}
        </Typography>
        <Grid
          container
          direction="row"
          justify="space-evenly"
          alignItems="center"
        >
          <Grid item>
            <TextField
              id="select-model-class"
              select
              label="Model Class"
              className={classes.textField}
              value={modelClass[0]}
              onChange={onHandleChange("modelClass")}
              SelectProps={{
                MenuProps: {
                  className: classes.menu,
                },
              }}
              helperText="Please select a model class"
              margin="normal"
            >
              {modelClass.map((item, i) => (
                <MenuItem value={item} key={item+i}>
                  {item}
                </MenuItem>
              ))}
            </TextField>              
          </Grid>
        </Grid>
      </React.Fragment>
    )
  }
}

export default withStyles(styles)(ModelClassSelect)