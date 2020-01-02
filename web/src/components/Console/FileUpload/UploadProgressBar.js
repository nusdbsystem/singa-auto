import React from "react"
import PropTypes from "prop-types"

import Typography from "@material-ui/core/Typography"
import Grid from "@material-ui/core/Grid"

import { Progress } from "react-sweet-progress"
import "react-sweet-progress/lib/style.css"

class UploadProgressBar extends React.Component {
  static propTypes = {
    percentCompleted: PropTypes.number.isRequired,
    fileName: PropTypes.string,
    dataset: PropTypes.string,
    formState: PropTypes.string,
  }

  render() {
    const { percentCompleted, fileName, formState, dataset } = this.props

    return (
      <React.Fragment>
        {formState !== "init" &&
          <Grid
            container
            direction="row"
            justify="flex-start"
            alignItems="center"
          >
            <Typography component="p">
              {fileName ? "Upload " + fileName : "No file chosen"}
            </Typography>
            <Progress
              theme={{
                error: {
                  symbol: " ",
                  trailColor: "pink",
                  color: "red",
                },
                active: {
                  symbol: "",
                  //trailColor: 'yellow',
                  color: "orange",
                },
                success: {
                  symbol: "",
                  //trailColor: 'lime',
                  color: "green",
                },
              }}
              percent={percentCompleted}
              status={formState === "loading" ? "active" : "success"}
            />
            {(formState === "idle") &&
              <Typography component="p">
                {dataset} created
              </Typography>
            }
          </Grid>
        }
      </React.Fragment>
    )
  }
}

export default UploadProgressBar
