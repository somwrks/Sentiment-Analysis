import * as tf from '@tensorflow/tfjs';

// Constants
const sequenceLength = 100; // Example sequence length
const vocabSize = 10000; // Example vocabulary size

// Tokenization and Vectorization Utility Function
function tokenizeAndVectorize(text: string): number[] {
  const words = text.split(' ');
  const paddedWords = words.slice(0, sequenceLength).concat(Array(sequenceLength - words.length).fill(''));
  const numericTokens = paddedWords.map((word) => hashFunction(word));
  return numericTokens.flat(); // Flatten tokens to a 1D array
}

// Hash Function
function hashFunction(word: string): number[] {
  return word.split('').map((char) => char.charCodeAt(0));
}

// Mock data from CSV for testing (replace with actual data loading)
const csvData = `88dfba044d," hey perez! good luck! well, i\`m still planning for a big present to be given for my mom! i love you mom!",good,positive,morning,0-20,Hungary,9660351,90530,107
40adafcc68," Can\`t go out this weekend, 12 hours of exams next week prevent it  Got my uni summerball next Sat though, should be good!","Can\`t go out this weekend, 12 hours of exams next week prevent it  Got my uni summerball next Sat though, should be good!",neutral,noon,21-30,Iceland,341243,100250,3
295d95bb65, naisee. is it bad that i can see lens flares all arond me while listening to it? XD cant really catch what they\`re saying tho,naisee. is it bad that i can see lens flares all arond me while listening to it? XD cant really catch what they\`re saying tho,neutral,night,31-45,India,1380004385,2973190,464
90273f6b75,"is up, slightly later than planned... gunna get this essay done 2day!!","is up, slightly later than planned... gunna get this essay done 2day!!",neutral,morning,46-60,Indonesia,273523615,1811570,151
ac04d1ec48, amen to that brotha!,amen to that brotha!,neutral,noon,60-70,Iran,83992949,1.63E+06,52
263460b27b, company policy. Has been for the last two places I\`ve worked at,company policy. Has been for the last two places I\`ve worked at,neutral,night,70-100,Iraq,40222493,434320,93
cf3b5b1730, won\`t need my hugs anymore,won\`t need my hugs anymore,negative,morning,0-20,Ireland,4937786,68890,72
090997a993, Congratulations on winning the Indie Award,Congratulations,positive,noon,21-30,Israel,8655535,21640,400
3706116a65,always makes bad decisions,always makes bad decisions,negative,night,31-45,Italy,60461826,294140,206
07c6f766e3,Eating.,Eating.,neutral,morning,46-60,Jamaica,2961167,10830,273
7908571329,_shines92 aww that sucks,sucks,negative,noon,60-70,Japan,126476461,364555,347`;

// Function to parse CSV data into arrays
function parseCsvData(csvData: string): { texts: string[]; labels: number[] } {
  const lines = csvData.split('\n');
  const texts: string[] = [];
  const labels: number[] = [];
  lines.forEach((line) => {
    const parts = line.split(',');
    const numericText = parts[1].replace(/"/g, '').split(' ').map(token => hashFunction(token));
    const stringText = numericText.map(token => token.join('')).join(' ');
    texts.push(stringText);
    labels.push(parts[3] === 'positive' ? 1 : 0);
  });
  return { texts, labels };
}
// Fetch Data Function (Mock implementation for testing)
async function fetchData(): Promise<{ texts: string[]; labels: number[] }> {
  console.log('Fetching data...');
  // Parse CSV data (replace with actual data loading)
  const { texts, labels } = parseCsvData(csvData);
  console.log('Fetching data... done');
  return { texts, labels };
}

// Train Model Function
// Train Model Function
async function trainModel(): Promise<tf.LayersModel> {
  console.log('Training model...');
  const { texts, labels } = await fetchData();

  // Tokenize and vectorize the texts
  const tokenizedTexts = texts.map(tokenizeAndVectorize);

  // Filter and pad the tokenized texts
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

  // Reshape and process the data
  const tensorizedTexts = tf.tensor(uniformPaddedTokenizedTexts, [uniformPaddedTokenizedTexts.length, sequenceLength], 'int32');
  const reshapedTensor = tf.reshape(tensorizedTexts, [-1, sequenceLength, 1]);

  // Create and compile the model
  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [sequenceLength, 1], units: 8, activation: 'relu' }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
  model.compile({ loss: 'binaryCrossentropy', optimizer: 'adam', metrics: ['accuracy'] });

  // Prepare labels and train the model
  const yTrain = tf.tensor1d(filteredLabels);
  await model.fit(reshapedTensor, yTrain, { epochs: 5 });

  console.log('Training model... done');

  return model;
}


// Sentiment Analysis Function
export async function sentimentAnalysisFunction(text: string): Promise<string> {
  console.log('Performing sentiment analysis...');
  const model = await trainModel(); // Assuming trainModel() properly handles tokenization and padding

  // Tokenize and vectorize the input text
  const numericText = tokenizeAndVectorize(text);

  // Pad the numericText if needed to match sequenceLength
  const paddedNumericText = numericText.concat(Array(sequenceLength - numericText.length).fill(0));

  // Create a tensor from the padded numericText
  const inputTensor = tf.tensor2d([paddedNumericText], [1, sequenceLength]);

  // Reshape the input tensor to match the model's input shape (if necessary)
  const reshapedInput = inputTensor.reshape([1, sequenceLength, 1]);

  // Perform prediction using the model
  const prediction = model.predict(reshapedInput) as tf.Tensor<tf.Rank>;
  const sentiment = prediction.dataSync()[0] > 0.5 ? 'positive' : 'negative';

  return sentiment;
}
