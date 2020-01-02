import React from 'react';
import PropTypes from 'prop-types';

import {useDropzone} from 'react-dropzone';

import { makeStyles } from '@material-ui/core/styles';
// File List
import AttachFileIcon from '@material-ui/icons/AttachFile';
import List from '@material-ui/core/List';
import ListSubheader from '@material-ui/core/ListSubheader';
import ListItem from '@material-ui/core/ListItem';
import ListItemAvatar from '@material-ui/core/ListItemAvatar';
import ListItemText from '@material-ui/core/ListItemText';
import Avatar from '@material-ui/core/Avatar';
import ListItemSecondaryAction from '@material-ui/core/ListItemSecondaryAction';
import IconButton from '@material-ui/core/IconButton';
import DeleteIcon from '@material-ui/icons/Delete';
import Typography from '@material-ui/core/Typography';
import Tooltip from '@material-ui/core/Tooltip';

const useStyles = makeStyles({
  root: {
    width: '100%',
    maxWidth: 360,
    backgroundColor: "#fff",
    marginLeft: 15
  }
})

// for file dropzone
const baseStyle = {
  width: "100%",
  maxWidth: 360,
  height: 100,
  borderWidth: 2,
  borderColor: '#666',
  borderStyle: 'dashed',
  borderRadius: 5,
  margin: "0 auto"
};
const activeStyle = {
  borderStyle: 'solid',
  borderColor: '#6c6',
  backgroundColor: '#eee'
};
const rejectStyle = {
  borderStyle: 'solid',
  borderColor: '#c66',
  backgroundColor: '#eee'
};

// React-Dropzone with hook
function FileDropzone(props) {
  const classes = useStyles();
  const {
    onCsvDrop,
    files,
    onRemoveCSV
  } = props

  const {
    //acceptedFiles,
    getRootProps,
    getInputProps,
    isDragAccept,
    isDragActive,
    isDragReject
  } = useDropzone({
    // Do something with the acceptedFiles
    onDrop: onCsvDrop
  });

  const reformatFileSize = fileSize => {
    if (fileSize < 1024) {
      return fileSize + " bytes"
    } else if (fileSize >= 1024 && fileSize < 1048576) {
      return (fileSize/1024).toFixed(2) + " kB"
    } else if (fileSize >= 1048576 && fileSize < 1073741824)
      return (fileSize/1048576).toFixed(2) + " MB"
  }
  
  const FileList = (
    <List
      subheader={<ListSubheader>CSV File:</ListSubheader>}
      className={classes.root}
    >
      {files.map(file => (
        <ListItem key={file.name}>
          <ListItemAvatar>
            <Avatar>
              <AttachFileIcon />
            </Avatar>
          </ListItemAvatar>
          <ListItemText
            primary={
              file.name.length > 25
                ? (
                  <Tooltip title={file.name}>
                    <span>{file.name.slice(0,20)+"..."}</span>
                  </Tooltip>
                )
                : file.name
            }
            secondary={reformatFileSize(file.size)}
          />
          <ListItemSecondaryAction>
            <IconButton
              aria-label="Delete"
              onClick={onRemoveCSV}  
            >
              <DeleteIcon />
            </IconButton>
          </ListItemSecondaryAction>
        </ListItem>
      ))}
    </List>
  )

  return (
    <section className="container">
      <div {...getRootProps({className: 'dropzone'})}>
        <input {...getInputProps()} />
        <p>Drag 'n' drop some files here, or click to select files</p>
      </div>
      <aside>
        {FileList}
      </aside>
    </section>
  );
}

FileDropzone.propTypes = {
  onCsvDrop: PropTypes.func.isRequired,
  files: PropTypes.array.isRequired,
  onRemoveCSV: PropTypes.func.isRequired,
}

export default FileDropzone
