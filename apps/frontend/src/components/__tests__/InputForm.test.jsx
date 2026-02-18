import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { vi, describe, expect, it, beforeEach } from 'vitest';
import InputForm from '../InputForm';

vi.mock('../../api/dataService', () => ({
  getUniqJobTitle: vi.fn(),
  retrainModel: vi.fn(),  
}));

import { getUniqJobTitle, retrainModel } from '../../api/dataService';

describe("InputForm", () => {
  beforeEach(() => {
    vi.clearAllMocks();

    getUniqJobTitle
    .mockResolvedValue({ value: ['Data Scientist', 'Data Engineer']});

    retrainModel
    .mockResolvedValue({ result: 'success', message: 'Retrain model successfully.'});
  });

  const baseProps = {
    getSubmitData: vi.fn(),
    handleInputFormSubmit: vi.fn(),
    setPredictResult: vi.fn(),
    addToast: vi.fn(),
    loadingFunc: false,
    setLoadingFunc: vi.fn(),
    setErrFunc: vi.fn(),
    dataAdded: false,
    setDataAdded: vi.fn(),
  }

  it('renders the forms', async () => {
    render(<InputForm {...baseProps}/>);

    expect(screen.getByText(/Salary Prediction/)).toBeInTheDocument();
    expect(screen.getByLabelText(/Age/)).toBeInTheDocument();
    expect(screen.getByLabelText(/Gender/)).toBeInTheDocument();
    expect(screen.getByLabelText(/Education level/)).toBeInTheDocument();
    expect(await screen.findByLabelText(/Job title/)).toBeInTheDocument();
    expect(screen.getByLabelText(/Years of experience/)).toBeInTheDocument();
    expect(document.querySelector('input[type=checkbox]')).toBeInTheDocument();
    expect(document.querySelector('label.form-check-label')).toBeInTheDocument();
    expect(document.querySelector('button.btn-link')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Predict Salary/})).toBeInTheDocument();
    expect(document.querySelector('button#ageYearModalTrigger')).toBeInTheDocument();
  });

  it('submit when inputs are valid', async () => {
    render(<InputForm {...baseProps} />);

    const ageSelect = screen.getByLabelText(/Age/);
    const genderSelect = screen.getByLabelText(/Gender/);
    const eduSelect = screen.getByLabelText(/Education level/);
    const jobSelect = await screen.findByLabelText(/Job title/);
    const yearSelect = screen.getByLabelText(/Years of experience/);
    const submitBtn = screen.getByRole('button', { name: /Predict Salary/});
    const checkbox = document.querySelector('input[type=checkbox]');

    fireEvent.change(ageSelect, { target: { value: '27' } });
    fireEvent.change(genderSelect, { target: { value: 'male' } });
    fireEvent.change(eduSelect, { target: { value: 'Master' } });
    fireEvent.change(jobSelect, { target: { value: 'Data Scientist' } });
    fireEvent.change(yearSelect, { target: { value: '7' } });
    fireEvent.click(checkbox);

    fireEvent.click(submitBtn);

    await waitFor(() => {
      expect(baseProps.getSubmitData).toHaveBeenCalledWith({
        age: '27',
        gender: 'male',
        education_level: 'Master',
        job_title: 'Data Scientist',
        years_of_experience: '7',
      });
      expect(baseProps.handleInputFormSubmit).toHaveBeenCalled();
    })
  });

  it('returns submit when inputs are invalid', async () => {
    render(<InputForm {...baseProps} />);

    const ageSelect = screen.getByLabelText(/Age/);
    const genderSelect = screen.getByLabelText(/Gender/);
    const eduSelect = screen.getByLabelText(/Education level/);
    const jobSelect = await screen.findByLabelText(/Job title/);
    const yearSelect = screen.getByLabelText(/Years of experience/);
    const checkbox = document.querySelector('input[type=checkbox]');

    const submitBtn = screen.getByRole('button', { name: /Predict Salary/});


    // age input invalid
    fireEvent.change(ageSelect, { target: { value: '' } });
    fireEvent.change(genderSelect, { target: { value: 'male' } });
    fireEvent.change(eduSelect, { target: { value: 'Master' } });
    fireEvent.change(jobSelect, { target: { value: 'Data Scientist' } });
    fireEvent.change(yearSelect, { target: { value: '7' } });
    fireEvent.click(checkbox);

    fireEvent.click(submitBtn);

    await waitFor(() => {
      expect(baseProps.handleInputFormSubmit).not.toHaveBeenCalled();
    })
  });

  it('triggers modal and return submit when age - yearE < 18', async () => {
    render(<InputForm {...baseProps} />);

    const formContainer = document.querySelector('form#InputForm');
    const ageSelect = screen.getByLabelText(/Age/);
    const genderSelect = screen.getByLabelText(/Gender/);
    const eduSelect = screen.getByLabelText(/Education level/);
    const jobSelect = await screen.findByLabelText(/Job title/);
    const yearSelect = screen.getByLabelText(/Years of experience/);
    const checkbox = document.querySelector('input[type=checkbox]');

    const modalTrigger = document.querySelector('button#ageYearModalTrigger');
    modalTrigger.click = vi.fn();

    fireEvent.change(genderSelect, { target: { value: 'male' } });
    fireEvent.change(eduSelect, { target: { value: 'Master' } });
    fireEvent.change(jobSelect, { target: { value: 'Data Scientist' } });
    fireEvent.click(checkbox);

    // age - yearE input invalid
    fireEvent.change(ageSelect, { target: { value: '18' } });
    fireEvent.change(yearSelect, { target: { value: '7' } });

    await waitFor(() => {
      expect(yearSelect.value).toBe('');
      expect(modalTrigger.click).toHaveBeenCalled();
      expect(formContainer).toHaveClass('was-validated');
      expect(baseProps.setPredictResult).toHaveBeenCalledWith(false);
    })
  });

  it('retrain button visiable when data added', async () => {
    render(<InputForm {...baseProps} dataAdded={true} />);
    expect(await screen.findByText('Retrain Model')).toBeInTheDocument();
  });

  it('calls retrainModel when clicking retrain button', async () => {
    render(<InputForm {...baseProps} dataAdded={true} />);
    const retrainBtn = await screen.findByText('Retrain Model');
    fireEvent.click(retrainBtn);

    await waitFor(() => {
      expect(baseProps.setLoadingFunc).toHaveBeenCalled();
      expect(baseProps.addToast)
      .toHaveBeenCalledWith('Retrain Model ...', 'warning');
    });

    await waitFor(() => {
      expect(baseProps.addToast)
      .toHaveBeenCalledWith('Retrain model successfully!', 'success');
      expect(baseProps.setLoadingFunc).toHaveBeenCalledWith(false);
    });
  });


  it('shows error when retrainModel fails', async () => {
    retrainModel.mockRejectedValue(new Error('Retrain model failed'));
    render(<InputForm {...baseProps} dataAdded={true} />);
    const retrainBtn = await screen.findByText('Retrain Model');
    fireEvent.click(retrainBtn);

    await waitFor(() => {
      expect(baseProps.setLoadingFunc).toHaveBeenCalledWith(true);
      expect(baseProps.setErrFunc).toHaveBeenCalledWith('Retrain model failed');
      expect(baseProps.addToast).toHaveBeenCalledWith("Retrain model failed!", "danger");
      expect(baseProps.setLoadingFunc).toHaveBeenCalledWith(false);
    });
  });

  it('console err message when job option fetching fail', async () => {
    getUniqJobTitle.mockRejectedValue(new Error('api failed'));
    render(<InputForm {...baseProps} />);
    const logEvent = vi.spyOn(console, 'log').mockImplementation(() => {});

    await waitFor(() => {
      expect(logEvent).toHaveBeenCalledWith(expect.any(Error))
    })
  });

  it('correctly disabled retrain and predict button when loading', async () => {
    render(<InputForm {...baseProps} dataAdded={true} loadingFunc={true} />);
    const submitBtn = screen.getByRole('button', { name: /Predict Salary/});
    const retrainBtn = await screen.findByText('Retrain Model');

    expect(submitBtn).toHaveClass('disabled');
    expect(retrainBtn).toHaveClass('disabled');


  });
});
