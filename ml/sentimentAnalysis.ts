import * as tf from '@tensorflow/tfjs';

// Constants
const sequenceLength = 100; // Example sequence length
const vocabSize = 10000; // Example vocabulary size

// Tokenization and Vectorization Utility Function
async function tokenizeAndVectorize(text: string): Promise<number[][]> {
  console.log('Tokenizing and vectorizing text...');
  if (text.length > sequenceLength * vocabSize) {
     throw new Error('Input text is too long.');
  }
  return new Promise((resolve) => {
     setTimeout(() => {
       const words = text.split(' ');
       const paddedWords = words.slice(0, sequenceLength).concat(Array(sequenceLength - words.length).fill(''));
       const numericTokens = paddedWords.map((word) => {
         const index = hashFunction(word) % vocabSize;
         const oneHotVector = new Array(vocabSize).fill(0);
         oneHotVector[index] = 1;
         return oneHotVector;
       });
       resolve(numericTokens);
     }, 1000);
  });
 }
 

// Fetch Data Function
async function fetchData(): Promise<{ texts: string[]; labels: number[] }> {
  console.log('Fetching data...');
  try {
    const response = await fetch('/dataset/train.csv');
    const data = await response.text();

    const texts: string[] = [];
    const labels: number[] = [];

    data.split('\n').forEach((row) => {
      const [text, sentiment] = row.split(',');
      texts.push(text);
      labels.push(sentiment === 'positive' ? 2 : sentiment === 'negative' ? 0 : 1);
    });
console.log("Fetching data... done")
    return { texts, labels };
  } catch (error) {
    console.error('Error fetching data:', error);
    throw new Error('Failed to fetch data.');
  }
}

// Train Model Function
async function trainModel(): Promise<tf.LayersModel> {
  console.log('Training model...');
  let tensor: tf.Tensor3D | undefined;
  try {
    const { texts, labels } = await fetchData();
    if (texts.length === 0 || labels.length === 0) {
      throw new Error('Empty data for training.');
    }

    const batchSize = 100; // Adjust batch size as needed
    const totalBatches = Math.ceil(texts.length / batchSize);

    const model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [sequenceLength, 1], units: 8, activation: 'relu' }));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));

    model.compile({ loss: 'categoricalCrossentropy', optimizer: 'adam', metrics: ['accuracy'] });

    for (let i = 0; i < totalBatches; i++) {
      const startIdx = i * batchSize;
      const endIdx = Math.min((i + 1) * batchSize, texts.length);
      const batchTexts = texts.slice(startIdx, endIdx);
      const batchLabels = labels.slice(startIdx, endIdx);

      const numericTokens = await Promise.all(batchTexts.map(tokenizeAndVectorize));

      // Flatten and reshape the numeric tokens to match expected input shape
      const reshapedBatchData = numericTokens.map(tokens => tf.tensor2d(tokens).transpose().reshape([sequenceLength, 1]));

      console.log(`Fitting model - Batch ${i + 1}/${totalBatches}`);
      await model.fit(reshapedBatchData, tf.oneHot(tf.tensor1d(batchLabels, 'int32'), 3), { epochs: 1 });

      tensor.dispose(); // Dispose of the tensor after each batch
    }

    model.summary();
    console.log("Training model... done");

    return model;
  } catch (error) {
    console.error('Error training model:', error);
    throw new Error('Failed to train model.');
  }
}



// Sentiment Analysis Function
export async function sentimentAnalysisFunction(text: string): Promise<string> {
  console.log('Performing sentiment analysis...');
  try {
    const model = await trainModel();
    const inputTensor = tf.tensor3d([await tokenizeAndVectorize(text)]).reshape([1, sequenceLength, 1]);
    console.log("inputtensor... done")

    console.log('Making prediction...');
    const prediction = model.predict(inputTensor) as tf.Tensor;
    const sentimentIndex = prediction.argMax().dataSync()[0];
console.log("making prediction... done")

    return sentimentIndex === 1 ? 'positive' : 'negative';
  } catch (error) {
    console.error('Error performing sentiment analysis:', error);
    throw new Error('Failed to perform sentiment analysis.');
  }
}

// Example hash function for demonstration purposes
function hashFunction(str: string): number {
  
  let hash = 0;
  if (str.length === 0) return hash;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = (hash << 5) - hash + char;
    hash = hash & hash; // Convert to 32bit integer
  }
  console.log("hasing function... done")
  return hash;
}
