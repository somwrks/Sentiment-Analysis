# Sentiment Analysis App

This is a Next.js application that performs sentiment analysis on text data using a TensorFlow.js model. The application is built with TypeScript and utilizes the TensorFlow.js library for training and evaluating the machine learning model.

## Overview

The project consists of the following components:

1. `pages/index.tsx`: The main page that allows users to input text and receive the sentiment analysis result.
2. `ml/sentimentAnalysis.ts`: This module contains the core logic for training the sentiment analysis model and performing predictions.

## Dependencies

The project requires the following dependencies:

- Next.js
- TensorFlow.js
- React
- TypeScript

## Setup

1. Clone the repository:

```bash
git clone https://github.com/your-repo/sentiment-analysis-app.git
```
2. Installing dependencies:
   ```bash
   cd sentiment-analysis-app
   npm install
   ```
4. Start the development server:
   ```bash
   npm run dev
   ```
   The application will be accessible at http://localhost:3000.

## Usage
  
  1. Open the application in your web browser.
  2. Enter some text in the input field.
  3. Click the "Analyze Sentiment" button.
  4. The sentiment analysis result (positive, negative, or neutral) will be displayed.

## Code Structure

- pages/index.tsx: This file contains the main page component, which includes the input field and displays the sentiment analysis result.

- ml/sentimentAnalysis.ts: This module is responsible for training the sentiment analysis model and performing predictions. It utilizes the TensorFlow.js library for machine learning tasks.

- public/dataset/: This directory contains the CSV files used for training and testing the sentiment analysis model.

## Deployment

  The Next.js application can be deployed to various hosting platforms, such as Vercel, Netlify, or a custom server.

## License

  This project is licensed under the MIT License.
