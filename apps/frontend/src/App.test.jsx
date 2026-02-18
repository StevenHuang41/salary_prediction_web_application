import { render, screen } from '@testing-library/react';
import App from './App';
import './App.css';
import { describe, it } from 'vitest';

describe('App', () => {
  it('reanders the main page', () => {
    render(<App />);
    const headline = screen.getByText(/Salary Prediction/i);
    expect(headline).toBeInTheDocument();
  });
});
