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


class DatasetSelect extends React.Component {
  static propTypes = {
    classes: PropTypes.object.isRequired,
    title: PropTypes.string,
    purpose: PropTypes.string.isRequired,
    datasetList: PropTypes.array,
    onHandleChange: PropTypes.func,
    selectedDataset: PropTypes.string.isRequired,
  }

  render() {
    const {
      classes,
      title,
      purpose,
      datasetList,
      selectedDataset,
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
              id="select-dataset"
              select
              label={`${purpose} dataset`}
              className={classes.textField}
              value={selectedDataset}
              onChange={onHandleChange(`selected${purpose}DS`)}
              helperText="Please select a dataset"
              margin="normal"
            >
              {datasetList.map((item, i) => (
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

export default withStyles(styles)(DatasetSelect)