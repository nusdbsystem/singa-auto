// the code is modified from https://benmccormick.org/2017/05/11/building-normal-curves-highcharts/

/* drop the left side of the equation. It is the normalization constant for the equation:
  it ensures that the total area under the curve equals 1,
  but doesn’t change the shape of the curve.
  Since we’re simply displaying the shape of the graph and care primarily
  about showing the range along the x axis,
  we can ignore it and instead use this function:
  const normalY = (x, mean, stdDev) => Math.exp((-0.5) * Math.pow((x - mean) / stdDev, 2));
*/
import _ from "lodash"

export const calculateGaussian = (mean, std) => {
  const normalY = (x, mean, stdDev) => Math.exp((-0.5) * Math.pow((x - mean) / stdDev, 2));

  const generatePoints = (lowerBound, upperBound) => {
    let min = lowerBound - 2 * std;
    let max = upperBound + 2 * std;
    let unit = (max - min) / 100;
    return _.range(min, max, unit);
  }

  let points = generatePoints(0, 1);

  //console.log("points: ", points)
  
  let seriesData = points.map(
    x => (
      [
        x,
        normalY(
          x,
          Math.min(
            Number(mean).toFixed(2),
            0.99
          ),
          Math.max(
            Number(std).toFixed(2),
            0.01
          )
        )
      ]
    )
  );

  return seriesData
}
