import * as tf from '@tensorflow/tfjs';
import * as fs from 'fs';
import csvParser from 'csv-parser';

const csvFilePath = '../sentimentalanalysis/dataset/train.csv';

const texts: string[] = [];
const labels: number[] = [];
fs.createReadStream(csvFilePath)
  .pipe(csvParser())
  .on('data', (row) => {
    texts.push(row.text);
    labels.push(row.sentiment === 'positive' ? 2 : (row.sentiment === 'negative' ? 0 : 1));
  })
  .on('end', async () => {
    try {
      const textsTensor = tf.tensor2d(texts.map(text => [text.length]), [texts.length, 1]);
      const labelsTensor = tf.oneHot(tf.tensor1d(labels, 'int32'), 3); // 3 classes: negative, neutral, positive

      const model = tf.sequential();
      model.add(tf.layers.dense({ inputShape: [1], units: 8, activation: 'relu' }));
      model.add(tf.layers.dense({ units: 3, activation: 'softmax' })); // 3 output units for 3 sentiment classes
      model.compile({ loss: 'categoricalCrossentropy', optimizer: 'adam', metrics: ['accuracy'] });

      await model.fit(textsTensor, labelsTensor, { epochs: 10 });

      await model.save('file://./models');
      console.log('Model saved successfully.');
    } catch (error) {
      console.error('An error occurred:', error);
    }
  });
