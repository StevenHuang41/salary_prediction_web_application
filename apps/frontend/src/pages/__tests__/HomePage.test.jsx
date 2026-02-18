import { render, screen } from '@testing-library/react';
import { vi, describe, expect, it, beforeEach } from 'vitest';
import HomePage from '../HomePage';

vi.mock('../../components/InputForm', () => ({
  default: ({ getSubmitData, handleInputFormSubmit }) => (
    <div>
      <button
        onClick={() => {
          getSubmitData({
            age: 39,
            gender: 'male',
            education_level: 'master',
            job_title: 'Data Scientist',
            years_of_experience: 3,
          });
          handleInputFormSubmit();
        }}
      >
        Predict Salary 
      </button>
    </div>
  )
}));

vi.mock('../../components/OutputSection', () => ({
  default: () => <div>OutputSection</div>
}));

vi.mock('../../components/ErrorPredict', () => ({
  default: ({ data }) => <div>ErrorPredict: {data}</div>
}));

vi.mock('../../components/LoadingResult', () => ({
  default: () => <div>Loading ...</div>
}));

vi.mock('../../components/Toast', () => ({
  default: ({ toasts }) => (
    <div data-testid="MyToast">
      {toasts.map((t) => (
        <div key={t.id}>{t.message}</div>
      ))}
    </div>
  )
}));

vi.mock('../../api/dataService', () => ({
  predictSalary: vi.fn()
}));

import { predictSalary } from '../../api/dataService';
import userEvent from '@testing-library/user-event';


describe('HomePage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('render HomePage components', () => {
    render(<HomePage />)
    
    expect(screen.getByText('Predict Salary')).toBeInTheDocument();
    expect(screen.getByTestId('MyToast')).toBeInTheDocument();
  });

  it(
    'show loading when form submitted, and disappear after loading',
    async () => {
      predictSalary.mockImplementation(async () => {
        await new Promise(res => setTimeout(res, 10));
        return {};
      });

      render(<HomePage />);
      const submitBtn = screen.getByText('Predict Salary');
      await userEvent.click(submitBtn);

      expect(await screen.findByText('Loading ...', {}, { timeout: 3000 })).toBeInTheDocument();
      expect(await screen.findByText('OutputSection')).toBeInTheDocument();
      expect(screen.queryByText('Loading ...')).not.toBeInTheDocument();
    }
  );

  it('show loading err when predictSalary has error', async () => {
    predictSalary.mockImplementation(async () => {
      throw new Error('predictSalary fail, backend error.');
    });

    render(<HomePage />);
    const submitBtn = screen.getByText('Predict Salary');
    await userEvent.click(submitBtn);

    expect(screen.queryByText(/predictSalary fail, backend error./))
    .toBeInTheDocument();

    expect(screen.queryByText('Loading ...')).not.toBeInTheDocument();
  });
});
