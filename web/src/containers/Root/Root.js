import PropTypes from "prop-types"
import React, { Fragment } from "react"
import * as actions from "./actions"
import { connect } from "react-redux"

import NotificationArea from "../../components/RootComponents/NotificationArea"

class Root extends React.PureComponent {
  static propTypes = {
    children: PropTypes.oneOfType([
      PropTypes.arrayOf(PropTypes.node),
      PropTypes.node,
    ]),
    notification: PropTypes.object,
    handleNotificationClose: PropTypes.func,
    onTryAutoSignup: PropTypes.func,
  }

  componentDidMount() {
    this.props.onTryAutoSignup()
  }

  render() {
    const { children, notification, handleNotificationClose } = this.props

    // console.log("reduxToken: ", this.props.reduxToken)

    return (
      <Fragment>
        <NotificationArea
          handleClose={handleNotificationClose}
          message={notification.message}
          open={notification.show}
        />
        {children}
      </Fragment>
    )
  }
}

const mapStateToProps = state => ({
  // reduxToken: state.Root.token,
  notification: state.Root.notification,
})

const mapDispatchToProps = {
  handleNotificationClose: actions.notificationHide,
  onTryAutoSignup: actions.authCheckState,
}

export default connect(mapStateToProps, mapDispatchToProps)(Root)
