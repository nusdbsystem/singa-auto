import React, {useMemo} from 'react';
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
  },
  aside:{
    width: '250px',
    marginLeft: 'auto',
    marginRight: 'auto',
  }
})

// for file dropzone
const baseStyle = {
  flex: 1,
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  padding: '20px',
  borderWidth: 2,
  borderRadius: 5,
  borderColor: '#eeeeee',
  borderStyle: 'dashed',
  backgroundColor: '#fafafa',
  color: '#263238',
  outline: 'none',
  transition: 'border .24s ease-in-out',
  width: '250px',
  marginLeft: 'auto',
  marginRight: 'auto'
};
const activeStyle = {
  borderStyle: 'solid',
  borderColor: '#2196f3',
  backgroundColor: '#eee'
};
const acceptStyle = {
  borderColor: '#00e676'
};
const rejectStyle = {
  borderStyle: 'solid',
  borderColor: '#ff1744',
  backgroundColor: '#eee'
};

// React-Dropzone with hook
function FileDropzone(props) {
  const classes = useStyles();
  const {
    onCsvDrop,
    files,
    onRemoveCSV,
    AcceptedMIMEtypes,
    MIMEhelperText,
    UploadType,
  } = props

  const {
    //acceptedFiles,
    getRootProps,
    getInputProps,
    isDragAccept,
    isDragActive,
    isDragReject
  } = useDropzone({
    // Note that the onDrop callback will
    // always be invoked regardless if the
    // dropped files were accepted or rejected.
    // If you'd like to react to a specific scenario,
    // use the onDropAccepted/onDropRejected props.
    onDropAccepted: onCsvDrop,
    multiple: false,
    // MIME type for zip
    // https://stackoverflow.com/questions/6977544/rar-zip-files-mime-type
    accept: AcceptedMIMEtypes,
  });

  const style = useMemo(() => ({
    ...baseStyle,
    ...(isDragActive ? activeStyle : {}),
    ...(isDragAccept ? acceptStyle : {}),
    ...(isDragReject ? rejectStyle : {})
  }), [
    isDragActive,
    isDragAccept,
    isDragReject
  ]);

  const reformatFileSize = fileSize => {
    if (fileSize < 1024) {
      return fileSize + " bytes"
    } else if (fileSize >= 1024 && fileSize < 1048576) {
      return (fileSize/1024).toFixed(2) + " kB"
    } else if (fileSize >= 1048576 && fileSize < 1073741824)
      return (fileSize/1048576).toFixed(2) + " MB"
  }
  
  const FileList = files.length === 0 ? (
    <List
      subheader={<ListSubheader>No file chosen</ListSubheader>}
      className={classes.root}
    ></List>
  )
  :  (
    <List
      subheader={<ListSubheader>{UploadType} File:</ListSubheader>}
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
    <>
      <div {...getRootProps({style})}>
        <input {...getInputProps()} />
        <p>Drag 'n' drop {UploadType} here, or click to select your file</p>
        <em>{MIMEhelperText}</em>
        <br />
        <br />
        <Typography variant="body1" gutterBottom align="center">
          {isDragAccept ? 'Drop' : 'Drag'} {UploadType} here...
        </Typography>
      </div>
      {isDragReject && <b>Unsupported file type...</b>}
      <aside className={classes.aside}>
        {FileList}
      </aside>
    </>
  );
}

FileDropzone.propTypes = {
  onCsvDrop: PropTypes.func.isRequired,
  files: PropTypes.array.isRequired,
  onRemoveCSV: PropTypes.func.isRequired,
  AcceptedMIMEtypes: PropTypes.string.isRequired,
  MIMEhelperText: PropTypes.string.isRequired,
  UploadType: PropTypes.string.isRequired,
}

export default FileDropzone
