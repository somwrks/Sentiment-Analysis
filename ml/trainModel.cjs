const fs = require('fs');
const csvParser = require('csv-parser');
const tf = require('@tensorflow/tfjs');

// Define the path to your CSV file
const csvFilePath = '../sentimentalanalysis/dataset/train.csv';

// Load and preprocess the dataset
const texts = [];
const labels = [];

fs.createReadStream(csvFilePath)
  .pipe(csvParser())
  .on('data', (row) => {
    texts.push(row.text);
    labels.push(row.sentiment === 'positive' ? 2 : row.sentiment === 'negative' ? 0 : 1);
  })
  .on('end', async () => {
    try {
      const textsTensor = tf.tensor2d(texts.map((text) => [text.length]), [texts.length, 1]);
      const labelsTensor = tf.oneHot(tf.tensor1d(labels, 'int32'), 3);

      const model = tf.sequential();
      model.add(tf.layers.dense({ inputShape: [1], units: 8, activation: 'relu' }));
      model.add(tf.layers.dense({ units: 3, activation: 'softmax' })); 
      model.compile({ loss: 'categoricalCrossentropy', optimizer: 'adam', metrics: ['accuracy'] });

      // Train the model
      await model.fit(textsTensor, labelsTensor, { epochs: 10 });

      await model.save('file:///C:/Users/AISHWARYA/Desktop/sentimentalanalysis/models');

      console.log('Model saved successfully.');
    } catch (error) {
      console.error('An error occurred:', error);
    }
  });
