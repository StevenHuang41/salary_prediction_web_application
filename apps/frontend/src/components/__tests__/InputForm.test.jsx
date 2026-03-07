import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { vi, describe, expect, it, beforeEach } from 'vitest';
import InputForm from '../InputForm';
import { getUniqJobTitle } from '../../api/dataService';

vi.mock('../../api/dataService', () => ({
  getUniqJobTitle: vi.fn(),
}));

describe("InputForm", () => {

  const baseProps = {
    onSubmit: vi.fn(),
    setPredictState: vi.fn(),
    setFormData: vi.fn(),
    isPredicting: false,
  }

  beforeEach(() => {
    vi.clearAllMocks();
    getUniqJobTitle.mockResolvedValue([
      "Data Scientist",
      "Data Engineer"
    ]);
  });

  it('renders the forms correctly', async () => {
    render(<InputForm {...baseProps}/>);

    expect(screen.getByText(/Salary Prediction/)).toBeInTheDocument();
    expect(screen.getByLabelText(/Age/)).toBeInTheDocument();
    expect(screen.getByLabelText(/Gender/)).toBeInTheDocument();
    expect(screen.getByLabelText(/Education level/)).toBeInTheDocument();

    expect(await screen.findByLabelText(/Job title/)).toBeInTheDocument();
    expect(screen.getByLabelText(/Years of experience/)).toBeInTheDocument();

    expect(document.querySelector('input[type=checkbox]')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Predict Salary/})).toBeInTheDocument();

    expect(document.querySelector('label.form-check-label')).toBeInTheDocument();
    expect(document.querySelector('button.btn-link')).toBeInTheDocument();
    expect(document.querySelector('button#ageYearModalTrigger')).toBeInTheDocument();
  });

  it("updates parent formData whenever inputs change (via useEffect)", async () => {
    render(<InputForm {...baseProps} />);
    const ageSelect = screen.getByLabelText(/Age/);
    fireEvent.change(ageSelect, { target: { value: '30' } });

    await waitFor(() => {
      expect(baseProps.setFormData).toHaveBeenCalledWith(expect.objectContaining({
        age: '30'
      }));
    });
  });

  it('calls onSubmit when inputs are valid', async () => {
    render(<InputForm {...baseProps} />);

    const ageSelect = screen.getByLabelText(/Age/);
    const genderSelect = screen.getByLabelText(/Gender/);
    const eduSelect = screen.getByLabelText(/Education level/);
    const jobSelect = await screen.findByLabelText(/Job title/);
    const yearSelect = screen.getByLabelText(/Years of experience/);
    const checkbox = document.querySelector('input[type=checkbox]');
    const submitBtn = screen.getByRole('button', { name: /Predict Salary/ });

    fireEvent.change(ageSelect, { target: { value: '27' } });
    fireEvent.change(genderSelect, { target: { value: 'male' } });
    fireEvent.change(eduSelect, { target: { value: 'Master' } });
    fireEvent.change(jobSelect, { target: { value: 'Data Scientist' } });
    fireEvent.change(yearSelect, { target: { value: '0' } });
    fireEvent.click(checkbox);
    fireEvent.click(submitBtn);

    expect(baseProps.onSubmit).toHaveBeenCalledTimes(1);
  });

  it('fails validation if yearE is invalid', async () => {
    render(<InputForm {...baseProps} />);

    const ageSelect = screen.getByLabelText(/Age/);
    const yearSelect = screen.getByLabelText(/Years of experience/);
    const submitBtn = screen.getByRole('button', { name: /Predict Salary/ });

    fireEvent.change(ageSelect, { target: { value: '20' } });
    fireEvent.change(yearSelect, { target: { value: '10' } });

    expect(screen.getByText(/The years of experience should not exceed 2/i))

    fireEvent.click(submitBtn);
    expect(ageSelect.value).toBe("20");
    expect(yearSelect.value).toBe("");
  });

  it('fails validation if inputs are invalid', async () => {
    render(<InputForm {...baseProps} />);

    const ageSelect = screen.getByLabelText(/Age/);
    const genderSelect = screen.getByLabelText(/Gender/);
    const eduSelect = screen.getByLabelText(/Education level/);
    const jobSelect = await screen.findByLabelText(/Job title/);
    const yearSelect = screen.getByLabelText(/Years of experience/);
    const checkbox = document.querySelector('input[type=checkbox]');
    const submitBtn = screen.getByRole('button', { name: /Predict Salary/});

    fireEvent.change(ageSelect, { target: { value: '' } });
    fireEvent.change(genderSelect, { target: { value: '' } });
    fireEvent.change(eduSelect, { target: { value: '' } });
    fireEvent.change(jobSelect, { target: { value: '' } });
    fireEvent.change(yearSelect, { target: { value: '' } });
    fireEvent.click(checkbox);
    fireEvent.click(submitBtn);

    expect(baseProps.onSubmit).not.toHaveBeenCalled();
    expect(screen.getByText(/Please select a valid age./))
    expect(screen.getByText(/Please select a gender./))
    expect(screen.getByText(/Please select an education level./))
    expect(screen.getByText(/Please select a job title./))
    expect(screen.getByText(/Please select a valid number./))
  });

  it('shows "Predicting ..." state when isPredicting = true', () => {
    render(<InputForm {...baseProps} isPredicting={true} />);

    const submitBtn = screen.getByTestId('predictSalaryBtn');
    expect(submitBtn).toBeDisabled();
    expect(submitBtn).toHaveTextContent(/Predicting .../)
  })

  it('handles API failure for job title', async () => {
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    getUniqJobTitle.mockRejectedValue(new Error('api failed'));

    render(<InputForm {...baseProps} />);

    await waitFor(() => {
      expect(consoleSpy).toHaveBeenCalledWith(expect.any(Error))
    })
    consoleSpy.mockRestore()
  });

  it('shows loading job title when receiving nothing from backend', () => {
    getUniqJobTitle.mockResolvedValue([]);
    render(<InputForm {...baseProps} />);
    expect(screen.getByText(/Loading options/))
  });
});
