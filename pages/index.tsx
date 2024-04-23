import React, { useState } from 'react';
import { NextPage } from 'next';
import { sentimentAnalysisFunction } from '../ml/sentimentAnalysis';

const Home: NextPage = () => {
  const [inputText, setInputText] = useState<string>('');
  const [sentimentResult, setSentimentResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInputText(e.target.value);
  };

  const handleFormSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    try {
      console.log(typeof inputText)
      const result = await sentimentAnalysisFunction(inputText);
      setSentimentResult(result);
      setError(null);
    } catch (err) {
      console.log(err)
      setError('An error occurred. Please try again.');
    }
  };

  return (
    <main className="flex max-h-screen flex-col items-center justify-between p-24">
      <form onSubmit={handleFormSubmit}>
        <textarea
          value={inputText}
          className='text-black'
          onChange={handleInputChange}
          placeholder="Enter text for sentiment analysis"
          rows={4}
          cols={50}
        />
        <br />
        <button type="submit">Analyze Sentiment</button>
      </form>
      {sentimentResult && <p>Sentiment: {sentimentResult}</p>}
      {error && <p style={{ color: 'red' }}>{error}</p>}
    </main>
  );
};

export default Home;
