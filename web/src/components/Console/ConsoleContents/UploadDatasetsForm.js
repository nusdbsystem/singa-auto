import React from "react"
import { bindActionCreators, compose } from "redux"
import PropTypes from "prop-types"
import { connect } from "react-redux"
import * as actions from "containers/Datasets/actions"


// Third part dependencies
//import { Form, Field } from "react-final-form"

//import { FormTextField, FormSwitchField } from "mui-form-fields"

import FileUpload from "../FileUpload/FileUpload"


class UploadDatasetsForm extends React.Component {
  static propTypes = {
    // classes: PropTypes.object.isRequired,
    postCreateDataset: PropTypes.func,
  }

  state = {
    selectedFile: null,
  }

  onSubmit = values => {
    console.log("submit values", values)
    // Dispatch actions
    if (this.state.fromLocal) {
      // for redux actions const postCreateDataset
      // = (name, task, file, dataset_url)
      // for API service postCreateDataset, + token
      this.props.postCreateDataset(
        values.name,
        "IMAGE_CLASSIFICATION",
        values.dataset[0]
      )
    } else {
      this.props.postCreateDataset(
        values.name,
        "IMAGE_CLASSIFICATION",
        undefined,
        values.dataset_url
      )
    }
    console.log("selectedFile: ", this.state.selectedFile)
  }


  render() {
    return (
      <div style={{ textAlign: "center" }}>
        <FileUpload />
      </div>
    )
  }
}

function mapDispatchToProps(dispatch) {
  // TODO: move this to container redux
  return bindActionCreators(
    { postCreateDataset: actions.postCreateDataset },
    dispatch
  )
}

export default compose(connect(null, mapDispatchToProps))(UploadDatasetsForm)
