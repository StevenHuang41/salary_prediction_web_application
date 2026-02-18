import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import ErrorPredict from '../ErrorPredict';

describe('ErrorPredict', () => {
  const errMessage = 'predict api fail';
  it('shows data message', () => {
    render(<ErrorPredict data={errMessage} />);
    expect(screen.getByText(errMessage)).toBeInTheDocument();
  });
});