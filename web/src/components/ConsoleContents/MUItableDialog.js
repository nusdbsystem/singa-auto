import React from 'react';
import PropTypes from "prop-types"
import Button from '@material-ui/core/Button';
import Dialog from '@material-ui/core/Dialog';
import DialogActions from '@material-ui/core/DialogActions';
import DialogContent from '@material-ui/core/DialogContent';
import DialogContentText from '@material-ui/core/DialogContentText';
import DialogTitle from '@material-ui/core/DialogTitle';


function AlertDialog(props) {
  const {
    open,
    handleClose,
    row,
    mode
  } = props

  const CustomizeDialog = (mode, row) => {
    switch (mode) {
      case "ListDatasets":
        return (
          <React.Fragment>
            <b>ID: </b>{row.id}<br />
            <b>Name: </b>{row.name}<br />
            <b>Task: </b>{row.task}<br />
            <b>Size: </b>{row.size_bytes} bytes<br />
            <b>Date created: </b>{row.datetime_created}<br />
            <b>Statistics: </b><br />
            {Object.keys(row.stat).map((keyName, i) =>(
              <li key={i}>{keyName}: {row.stat[keyName]}</li>
            ))}
          </React.Fragment>
        )
      case "ListModels":
        return (
          <React.Fragment>
            <b>ID: </b>{row.id}<br />
            <b>Name: </b>{row.name}<br />
            <b>Task: </b>{row.task}<br />
            <b>Date created: </b>{row.datetime_created}<br />
            <b>Dependencies: </b><br />
            {Object.keys(row.dependencies).map((keyName, i) =>(
              <li key={i}>{keyName}: {row.dependencies[keyName]}</li>
            ))}
          </React.Fragment>
        )
      default:
        return (
          <b>No data to display</b>
        )
    }
  }

  return (
    <div>
      <Dialog
        open={open}
        onClose={handleClose}
        aria-labelledby="alert-dialog-title"
        aria-describedby="alert-dialog-description"
      >
        <DialogTitle id="alert-dialog-title">
          Dataset {Object.keys(row).length !== 0 && row.id.slice(0, 8)}
        </DialogTitle>
        <DialogContent>
          <DialogContentText id="alert-dialog-description">
            {Object.keys(row).length !== 0 &&
              CustomizeDialog(mode, row)
            }
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleClose} color="primary" autoFocus>
            Close
          </Button>
        </DialogActions>
      </Dialog>
    </div>
  );
}

AlertDialog.propTypes = {
  open: PropTypes.bool.isRequired,
  handleClose: PropTypes.func.isRequired,
  row: PropTypes.object.isRequired,
  mode:PropTypes.string.isRequired,
}

export default AlertDialog

