import * as tf from '@tensorflow/tfjs';

const sequenceLength = 100;

// Tokenization and vectorization function
function tokenizeAndVectorize(text: string): number[] {
  const words = text.toLowerCase().split(' '); // Convert to lowercase for consistency
  const paddedWords = words.slice(0, sequenceLength).concat(Array(sequenceLength - words.length).fill(''));
  const numericTokens = paddedWords.map((word) => hashFunction(word));
  return numericTokens.flat();
}

// Hash function
function hashFunction(word: string): number[] {
  return word.split('').map((char) => char.charCodeAt(0));
}

async function readCsvData(url: string): Promise<string> {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error('Failed to fetch CSV file');
    }
    const csvData = await response.text();
    return csvData;
  } catch (error) {
    console.error('Error reading CSV file:', error);
    throw error;
  }
}

function parseCsvData(csvData: string): { texts: string[]; labels: number[] } {
  const lines = csvData.split('\n');
  const texts: string[] = [];
  const labels: number[] = [];
  lines.forEach((line) => {
    const parts = line.split(',');
    console.log('Parts:', parts);
    if (parts.length >= 4) {
      const numericText = parts[1].replace(/"/g, '').split(' ').map(token => tokenizeAndVectorize(token));
      const stringText = numericText.map(token => token.join('')).join(' ');
      texts.push(stringText);
      labels.push(parts[3] === 'positive' ? 1 : 0);
    } else {
      console.log('Invalid line format:', line);
    }
  });
  return { texts, labels };
}

async function fetchData(url: string): Promise<{ texts: string[]; labels: number[] }> {
  console.log('Fetching data...');
  const csvData = await readCsvData(url);
  const { texts, labels } = parseCsvData(csvData);
  console.log('Fetching data... done');
  return { texts, labels };
}

async function trainModel(url: string): Promise<tf.LayersModel> {
  console.log('Training model...');
  const { texts, labels } = await fetchData(url);

  const tokenizedTexts = texts.map(tokenizeAndVectorize);

  const filteredTexts = tokenizedTexts.filter((tokens) => tokens.length <= sequenceLength);
  const filteredLabels = labels.filter((_, index) => tokenizedTexts[index].length <= sequenceLength);
  const paddedTokenizedTexts = filteredTexts.map((tokens) => {
    const paddingLength = sequenceLength - tokens.length;
    return tokens.concat(Array(paddingLength).fill(0));
  });
  const uniformPaddedTokenizedTexts = paddedTokenizedTexts.map((tokens) => {
    if (tokens.length !== sequenceLength) {
      return tokens.concat(Array(sequenceLength - tokens.length).fill(0));
    }
    return tokens;
  });

  const tensorizedTexts = tf.tensor(uniformPaddedTokenizedTexts, [uniformPaddedTokenizedTexts.length, sequenceLength], 'int32');
  const reshapedTensor = tf.reshape(tensorizedTexts, [-1, sequenceLength, 1]);

  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [sequenceLength, 1], units: 8, activation: 'relu' }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dropout({ rate: 0.2 }));
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

  model.compile({ loss: 'binaryCrossentropy', optimizer: 'adam', metrics: ['accuracy'] });

  const yTrain = tf.tensor1d(filteredLabels);
  await model.fit(reshapedTensor, yTrain, { epochs: 10 });

  console.log('Training model... done');

  return model;
}


const sentimentThresholds = {
  positive: 0.003, 
  neutral: 0.001,
  negative: 0.005,
};

export async function sentimentAnalysisFunction(text: string): Promise<string> {
  console.log('Performing sentiment analysis...');
  const csvUrl = '/dataset/train.csv';
  const model = await trainModel(csvUrl);

  const numericText = tokenizeAndVectorize(text);

  const paddedNumericText = numericText.concat(Array(sequenceLength - numericText.length).fill(0));

  const inputTensor = tf.tensor2d([paddedNumericText], [1, sequenceLength]);

  const reshapedInput = inputTensor.reshape([1, sequenceLength, 1]);

  const prediction = model.predict(reshapedInput) as tf.Tensor<tf.Rank>;
  console.log(prediction.dataSync()[0]);
  
  const positiveThreshold = sentimentThresholds.positive;
  const neutralThreshold = sentimentThresholds.neutral;
  const negativeThreshold = sentimentThresholds.negative;
  
  const sentimentValue = prediction.dataSync()[0];
  
  let sentiment: string;
  if (sentimentValue > positiveThreshold) {
    sentiment = 'positive';
  } else if (sentimentValue > neutralThreshold) {
    sentiment = 'neutral';
  } else if (sentimentValue > negativeThreshold) {
    sentiment = 'negative';
  } else {
    sentiment = 'neutral'; 
  }

  return sentiment;
}
