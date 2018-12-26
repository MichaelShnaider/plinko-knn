const outputs = [];

// const k = 3; // Our k-sized cluster
const numberOfFeaturesToConsier = 1; // Change between 1 and 3

function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {
  // Ran every time a balls drops into a bucket
  outputs.push([dropPosition, bounciness, size, bucketLabel]);

  // console.log(outputs);
}

function runAnalysis() {
  const testSetSize = 100;
  const [testSet, trainingSet] = splitDataset(normalizeFeatures(outputs, numberOfFeaturesToConsier), testSetSize); // Get the test/training set. args: (dataset, test size)

  _.range(1,15).forEach((k) => {
    const accuracy = _.chain(testSet)
      .filter(testPoint => knn(trainingSet, _.initial(testPoint), k) === _.last(testPoint))
      .size()
      .divide(testSetSize)
      .value()
    console.log('The accuracy is: ', accuracy, ' for k=', k);
  });

  console.log('Objective Feature Selection---------');
  _.range(0,3).forEach((featureNumber) => {
    const singleFeatureData = outputs.map((row) => [row[featureNumber], _.last(row)]);
    const [testSet, trainingSet] = splitDataset(normalizeFeatures(singleFeatureData, 1), testSetSize); // Get the test/training set. args: (dataset, test size)
    const accuracy = _.chain(testSet)
      .filter(testRow => knn(trainingSet, _.initial(testRow), 10) === _.last(testRow))
      .size()
      .divide(testSetSize)
      .value();

    console.log('The accuracy for feature #'+featureNumber, 'is:', accuracy);
  });
}

// pointA/B: [f1, f2, f3, ...]
function distance(pointA, pointB) { // Distance from our prediction point to the passed in point.
  // Calculates Euclidean distance
  return _.chain(pointA)
    .zip(pointB) // Creates an array of [[x1, x2, x3, ...], [y1, y2, y3, ...]]
    .map(([a, b]) => (a-b) ** 2)
    .sum() ** 0.5
}

// data is: [f1, f2, f3][]
// point is: [f1, f2, f3, label]
// k is a number.
function knn(trainingData, point, k) {
  return _.chain(trainingData) // Allows us to chain lodash commands
    .map(row => [distance(_.initial(row), point), _.last(row)]) // Reformats the data: [distance from point, bucket entered][]
    .sortBy(row => row[0]) // Sorts from least to greatest by first element, so that the elements closest to our dropping point are the first elements of the array.
    .slice(0, k) // Remove all rows after the kth row. (contains our cluster)
    .countBy(row => row[1]) // Returns a dict with the key being the value passed in as a string and the value being the number of times that value occurred.
    .toPairs() // Converts a regular 1-layer dictionary into [[key, value][]]
    .sortBy(row => row[1]) // Sorts by value returned, row[1] is the number of times the bucket was accessed, so sorted by most common bucket, from least to greatest
    .last() // Gets most common bucket
    .first() // Gets bucket number (string)
    .parseInt() // Converts to integer
    .value() // Ends lodash chain and gets value.
}

// Data: [f1, f2, f3, label][]
function normalizeFeatures(data, numOfFeatures) {
  // We never want to modify the original dataset so we deep copy first.
  const clonedData = _.cloneDeep(data);

  // Loop through each feature
  for (let featureIndex = 0; featureIndex < numOfFeatures; featureIndex++) {
    // Gets the full column
    const column = clonedData.map((row) => row[featureIndex]);
    const min = _.min(column);
    const max = _.max(column);
    // console.log('Col: ', column, 'MIN/MAX', min, max);

    // Loop through each row
    for (let rowIndex = 0; rowIndex < column.length; rowIndex++) {
      const featureVal = clonedData[rowIndex][featureIndex];
      // Replaced current feature with normalized feature.
      clonedData[rowIndex][featureIndex] = (featureVal - min) / (max - min);
    }
  }
  return clonedData;
}


function splitDataset(data, testCount) {
  const shuffled = _.shuffle(data); // Shuffle the dataset

  const testSet = _.slice(shuffled, 0, testCount); // Take everything from 0 to testCount
  const trainingSet = _.slice(shuffled, testCount); // Take everything from 0 to end of array

  return [testSet, trainingSet];
}

