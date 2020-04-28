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
  textField: {
    marginLeft: theme.spacing(1),
    marginRight: theme.spacing(1),
    width: 250,
  },
})


class ModelSelect extends React.Component {
  static propTypes = {
    classes: PropTypes.object.isRequired,
    title: PropTypes.string,
    modelList: PropTypes.array,
    onHandleChange: PropTypes.func,
    selectedModel: PropTypes.string.isRequired,
  }

  render() {
    const {
      classes,
      title,
      modelList,
      selectedModel,
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
              id="select-model"
              select
              label="Select Model"
              className={classes.textField}
              value={selectedModel}
              onChange={onHandleChange("selectedModel")}
              helperText="Please select a model"
              margin="normal"
            >
              {modelList.map((item, i) => (
                <MenuItem value={item.value} key={item.value+i}>
                  {item.label}
                </MenuItem>
              ))}
            </TextField>              
          </Grid>
        </Grid>
      </React.Fragment>
    )
  }
}

export default withStyles(styles)(ModelSelect)