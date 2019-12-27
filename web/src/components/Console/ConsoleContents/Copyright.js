import React from 'react';
import Typography from '@material-ui/core/Typography';
import Link from '@material-ui/core/Link';

function Copyright() {
  return (
    <Typography variant="body2" color="textSecondary" align="center">
      {'Copyright © '}
      <Link color="inherit" href="https://github.com/easyfan327/rafiki_panda_dev/">
        Panda-dev
      </Link>{' '}
      {new Date().getFullYear()}
    </Typography>
  );
}

export default Copyright