// ml/sentimentAnalysis.ts

import * as tf from '@tensorflow/tfjs';

let model: tf.LayersModel | null = null;

// Load the TensorFlow.js model
async function loadModel() {
    model = await tf.loadLayersModel('file://./models/model.json');
}

// Perform sentiment analysis using the loaded model
export async function sentimentAnalysisFunction(text: string): Promise<string> {
    if (!model) {
        await loadModel();
    }

    if (model) {
        const input = tf.tensor2d([[text]]);
        const prediction = model.predict(input) as tf.Tensor;
        const sentimentIndex = prediction.argMax().dataSync()[0];
        return sentimentIndex === 1 ? 'positive' : 'negative';
    } else {
        throw new Error('Model not loaded.');
    }
}
