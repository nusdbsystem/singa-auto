export default function getPlotDetails(plot, metrics) {
  const seriesByName = {}
  for (const plotMetric of plot.metrics) {
    seriesByName[plotMetric] = {
      data: [],
      name: plotMetric,
    }
  }
  const xAxis = plot.x_axis || "time"

  for (const metric of metrics) {
    // Check if x axis value exists
    if (!(xAxis in metric)) {
      continue
    }

    // For each of plot's y axis metrics, push the [x, y] to data array
    for (const plotMetric of plot.metrics) {
      if (!(plotMetric in metric)) {
        continue
      }

      // Push x axis value to data array
      seriesByName[plotMetric].data.push([metric[xAxis], metric[plotMetric]])
    }
  }

  const plotOption = {
    title: plot.title,
    xAxis: {
      // eslint-disable-next-line
      type: xAxis == "time" ? "time" : "number",
      name: xAxis,
    },
  }

  return { series: Object.values(seriesByName), plotOption }
}
