import React from 'react';
import PropTypes from 'prop-types';

import { makeStyles } from '@material-ui/core/styles';
import Table from '@material-ui/core/Table';
import TableBody from '@material-ui/core/TableBody';
import TableCell from '@material-ui/core/TableCell';
import TableContainer from '@material-ui/core/TableContainer';

import TablePagination from '@material-ui/core/TablePagination';
import TableRow from '@material-ui/core/TableRow';

import MoreVertIcon from '@material-ui/icons/MoreVert';

import IconButton from '@material-ui/core/IconButton';
import Typography from '@material-ui/core/Typography';

import {
  getComparator,
  stableSort,
} from "./MUItableUtils"

import EnhancedTableHead from "./MUItableHead"

import MUItableDialog from "./MUItableDialog"

var moment = require('moment');

const useStyles = makeStyles(theme => ({
  root: {
    width: '100%',
  },
  table: {
    minWidth: 750,
  },
  visuallyHidden: {
    border: 0,
    clip: 'rect(0 0 0 0)',
    height: 1,
    margin: -1,
    overflow: 'hidden',
    padding: 0,
    position: 'absolute',
    top: 20,
    width: 1,
  },
}));

function EnhancedTable(props) {
  const classes = useStyles();
  const {
    headCells,
    rows,
    mode,
  } = props
  const [order, setOrder] = React.useState('asc');
  const [orderBy, setOrderBy] = React.useState('UploadedAt');
  const [page, setPage] = React.useState(0);
  const [rowsPerPage, setRowsPerPage] = React.useState(5);
  const [selected, setSelected] = React.useState({});
  const [open, setOpen] = React.useState(false);

  const handleClose = () => {
    setOpen(false);
    setSelected({})
  };

  const handleViewMoreClick = (event, row) => {
    console.log("row OBJECT selected: ", row)
    setOpen(true);
    setSelected(row)
  };

  const handleRequestSort = (event, property) => {
    const isAsc = orderBy === property && order === 'asc';
    setOrder(isAsc ? 'desc' : 'asc');
    setOrderBy(property);
  };

  const handleChangePage = (event, newPage) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = event => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  const emptyRows = rowsPerPage - Math.min(rowsPerPage, rows.length - page * rowsPerPage);

  const CustomizeRows = (mode, row) => {
    switch (mode) {
      case "ListDatasets":
        return (
          <>
            <TableCell align="left">{row.name}</TableCell>
            <TableCell align="left">{row.task}</TableCell>
            <TableCell align="left">{row.size_bytes}</TableCell>
            <TableCell align="left">{moment(row.datetime_created).calendar()}</TableCell>
          </>
        )
      case "ListModels":
        return (
          <>
            <TableCell align="left">{row.name}</TableCell>
            <TableCell align="left">{row.task}</TableCell>
            <TableCell align="left">{JSON.stringify(row.dependencies)}</TableCell>
            <TableCell align="left">{moment(row.datetime_created).calendar()}</TableCell>
          </>
        )
      default:
        return (
          <TableCell colSpan={6} />
        )
    }
  }

  return (
    <div className={classes.root}>
      <TableContainer>
        <Table
          className={classes.table}
          aria-labelledby="tableTitle"
          size={'medium'}
          aria-label="enhanced table"
        >
          <EnhancedTableHead
            classes={classes}
            order={order}
            orderBy={orderBy}
            onRequestSort={handleRequestSort}
            headCells={headCells}
          />
          <TableBody>
            {stableSort(rows, getComparator(order, orderBy))
              .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
              .map((row, index) => {
                return (
                  <TableRow
                    hover
                    role="checkbox"
                    tabIndex={-1}
                    key={row.id+index}
                  >
                    <TableCell component="th" id={row.id} scope="row">
                      <Typography variant="overline" display="block" gutterBottom>
                        {row.id.slice(0, 8)}
                      </Typography>
                    </TableCell>
                    {CustomizeRows(mode, row)}
                    <TableCell padding="checkbox">
                      <IconButton
                        onClick={event => handleViewMoreClick(event, row)}
                        aria-label="show more"
                      >
                        <MoreVertIcon />
                      </IconButton>
                    </TableCell>
                  </TableRow>
                );
              })}
            {emptyRows > 0 && (
              <TableRow style={{ height:  53 * emptyRows }}>
                <TableCell colSpan={6} />
              </TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>
      <TablePagination
        rowsPerPageOptions={[5, 10, 25]}
        component="div"
        count={rows.length}
        rowsPerPage={rowsPerPage}
        page={page}
        onChangePage={handleChangePage}
        onChangeRowsPerPage={handleChangeRowsPerPage}
      />
      <MUItableDialog
        open={open}
        handleClose={handleClose}
        row={selected}
        mode={mode}
      />
    </div>
  );
}

EnhancedTable.propTypes = {
  headCells: PropTypes.array.isRequired,
  rows: PropTypes.array.isRequired,
  mode: PropTypes.string.isRequired,
}

export default EnhancedTable
