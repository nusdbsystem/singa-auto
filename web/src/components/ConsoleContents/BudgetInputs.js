import React from 'react';

import { makeStyles } from '@material-ui/core/styles';
import Grid from '@material-ui/core/Grid';
import Typography from '@material-ui/core/Typography';
import Input from '@material-ui/core/Input';


const useStyles = makeStyles({
  root: {
    width: 250,
  },
  input: {
    width: 82,
  },
});

export default function BudgetInputs(props) {
  const classes = useStyles();

  const {
    title,
    value_time_hours,
    value_gpu_count,
    value_model_trial_count,
  } = props

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
        <div className={classes.root}>
          <Grid container spacing={2} alignItems="center">
            <Grid item>
              TIME_HOURS
            </Grid>
            <Grid item>
              <Input
                className={classes.input}
                value={value_time_hours}
                margin="dense"
                onChange={props.onHandleChange("Budget_TIME_HOURS")}
                inputProps={{
                  step: 0.1,
                  min: 0.1,
                  max: 100,
                  type: 'number',
                  'aria-labelledby': 'input-slider',
                }}
              />
            </Grid>
          </Grid>

          <Grid container spacing={2} alignItems="center">
            <Grid item>
              GPU_COUNT
            </Grid>
            <Grid item>
              <Input
                className={classes.input}
                value={value_gpu_count}
                margin="dense"
                onChange={props.onHandleChange("Budget_GPU_COUNT")}
                inputProps={{
                  step: 1,
                  min: 0,
                  max: 100,
                  type: 'number',
                  'aria-labelledby': 'input-slider',
                }}
              />
            </Grid>
          </Grid>

          <Grid container spacing={2} alignItems="center">
            <Grid item>
              MODEL_TRIAL_COUNT
            </Grid>
            <Grid item>
              <Input
                className={classes.input}
                value={value_model_trial_count}
                margin="dense"
                onChange={props.onHandleChange("Budget_MODEL_TRIAL_COUNT")}
                inputProps={{
                  step: 1,
                  min: -1,
                  max: 1000,
                  type: 'number',
                  'aria-labelledby': 'input-slider',
                }}
              />
            </Grid>
          </Grid>
        </div>
      </Grid>
    </React.Fragment>
  );
}
