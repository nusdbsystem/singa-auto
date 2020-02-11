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
    row
  } = props

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
}

export default AlertDialog

