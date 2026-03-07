import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { vi, describe, expect, it, beforeEach } from 'vitest';
import OutputSection from '../OutputSection';
import userEvent from '@testing-library/user-event';

vi.mock('../../api/dataService', () => ({
  fetchSalaryHistPlot: vi.fn(),
  fetchSalaryBoxPlot: vi.fn(),
  addData: vi.fn(),
}));

import {
  fetchSalaryHistPlot,
  fetchSalaryBoxPlot,
  addData,
} from '../../api/dataService';

vi.mock('../MyCarousel', () => ({
  default: ({images, alts}) => (
    <div data-testid="MyCarousel">
      {images.map((image, idx) => (
        <img src={image} alt={alts[idx]} />
      ))}
    </div>
  )
}));

const baseProps = {
  dataFromForm: {
    age: 27,
    gender: 'male',
    education_level: 'master',
    job_title: 'Data Scientist',
    years_of_experience: 2,
  },
  predictData: {
    salary: 120000,
    model_name: 'HGBR',
    mse: 8000000,
    mae: 3000,
    rmse: 2800,
    n_train: 8000,
    n_test: 200
  },
  addToast: vi.fn(),
  isTraining: false,
  onRetrain: vi.fn(),
  onReset: vi.fn(),
  setDBChanged: vi.fn(),
  showRetrainBtn: false,
};

beforeEach(() => {
  vi.clearAllMocks();
  fetchSalaryHistPlot.mockResolvedValue('hist-url')
  fetchSalaryBoxPlot.mockResolvedValue('box-url')
});

describe('OutputSection', () => {
  it('renders output components and hides detail by default', async () => {
    render(<OutputSection {...baseProps} />);

    const input = document.querySelector('input#predict-input')
    expect(input).toBeInTheDocument();

    expect(screen.getByText(/Model/)).toBeInTheDocument();
    expect(screen.getByText(/MAE/)).toBeInTheDocument();

    const seeDetailBtn = screen.getByText('see detail');
    expect(seeDetailBtn).toBeInTheDocument();
    expect(seeDetailBtn).toHaveClass('btn-outline-secondary');

    expect(await screen.findByTestId('MyCarousel')).toBeInTheDocument();

    await waitFor(() => {
      expect(fetchSalaryHistPlot).toHaveBeenCalled();
      expect(fetchSalaryBoxPlot).toHaveBeenCalled();
    });

    // should not show when see detail is false
    expect(document.querySelector('input.form-range')).not.toBeInTheDocument();
    expect(seeDetailBtn).not.toHaveClass('btn-secondary');
    expect(screen.queryByText(/Model Name/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Mean Absolute Error/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Root Mean Square Error/)).not.toBeInTheDocument();
    // expect(screen.getByText("Train size:")).not.toBeInTheDocument();
    // expect(screen.getByText(/Test size/)).not.toBeInTheDocument();
    expect(screen.queryByText('Reset Database')).not.toBeInTheDocument();
    // expect(await screen.findByAltText('Salary Histogram Plot')).not.toBeInTheDocument();
    // expect(await screen.findByAltText('Salary Box Plot')).not.toBeInTheDocument();
  });

  it('renders output components when see detail is true', async () => {
    render(<OutputSection {...baseProps} showDetail={true} />);


    const seeDetailBtn = screen.getByText('see detail');
    expect(seeDetailBtn).toBeInTheDocument();
    fireEvent.click(seeDetailBtn);

    expect(screen.queryByRole('slider')).toBeInTheDocument();
    expect(screen.getByText(/Model Name/)).toBeInTheDocument();
    expect(screen.getByText(/Mean Absolute Error/)).toBeInTheDocument();
    expect(screen.getByText(/Root Mean Square Error/)).toBeInTheDocument();
    expect(screen.getByText(/Train size/)).toBeInTheDocument();
    expect(screen.getByText(/Test size/)).toBeInTheDocument();
    expect(screen.getByText('Reset Database')).toBeInTheDocument();
    expect(await screen.findByAltText('Salary Histogram Plot')).toBeInTheDocument();
    expect(await screen.findByAltText('Salary Box Plot')).toBeInTheDocument();

    expect(screen.queryByTestId('MyCarousel')).not.toBeInTheDocument();
  });

  it('updates salary via text input and triggers plots fetch', async () => {
    const user = userEvent.setup();
    render(<OutputSection {...baseProps} />);

    const input = screen.getByRole('textbox');

    await user.clear(input);
    await user.type(input, '130000');

    expect(input.value).toBe('130000');

    await waitFor(() => {
      expect(fetchSalaryHistPlot).toHaveBeenCalledWith(130000);
      expect(fetchSalaryBoxPlot).toHaveBeenCalledWith(130000);
    });
  });

  it('logs errors when fetching images api fails', async () => {
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    fetchSalaryHistPlot.mockRejectedValue(new Error('api fails'))
    const user = userEvent.setup()

    render(<OutputSection {...baseProps} />);

    const input = screen.getByRole('textbox');
    await user.clear(input);
    await user.type(input, '130000');


    await waitFor(() => {
      expect(consoleSpy).toHaveBeenCalledWith(new Error('api fails'));
    });
    consoleSpy.mockRestore()
  });

  it('shows "Return Input" and "Add Data" when salary is changed', async () => {
    const user = userEvent.setup();
    render(<OutputSection {...baseProps} />);

    fireEvent.click(screen.getByText('see detail'));

    const input = screen.getByRole('textbox');
    await user.clear(input);
    await user.type(input, '150000');

    expect(screen.getByText('Return Input')).toBeInTheDocument();
    expect(screen.getByText('Add Data')).toBeInTheDocument();

    await user.click(screen.getByText('Return Input'));
    expect(input.value).toBe('120,000');
  });

  it('calls addData and addToast when "Add Data" is clicked', async () => {
    const user = userEvent.setup();
    addData.mockResolvedValueOnce({});

    render(<OutputSection {...baseProps} />);
    fireEvent.click(screen.getByText('see detail'));

    const input = screen.getByRole('textbox');
    await user.type(input, '150000'); // Change value to make "Add Data" appear

    const addBtn = screen.getByText('Add Data');
    await user.click(addBtn);

    expect(addData).toHaveBeenCalled();
    expect(baseProps.setDBChanged).toHaveBeenCalledWith(true);
    expect(baseProps.addToast).toHaveBeenCalledWith("Data added successfully!", "success");
  });

  it('catches addData error and show toast error message', async () => {
    const user = userEvent.setup();
    addData.mockRejectedValue(new Error('api fails'));

    render(<OutputSection {...baseProps} />);
    fireEvent.click(screen.getByText('see detail'));

    const input = screen.getByRole('textbox');
    await user.type(input, '150000'); // Change value to make "Add Data" appear

    const addBtn = screen.getByText('Add Data');
    await user.click(addBtn);

    expect(addData).toHaveBeenCalled();
    expect(baseProps.addToast).toHaveBeenCalledWith("Failed to add data", "danger");
  });

  it('calls onReset when Reset Database is clicked', async () => {
    const user = userEvent.setup();
    render(<OutputSection {...baseProps} />);
    fireEvent.click(screen.getByText('see detail'));

    const resetBtn = screen.getByText('Reset Database');
    await user.click(resetBtn);

    expect(baseProps.onReset).toHaveBeenCalled();
  });

  it('updates salary text input when the range slider moves', async () => {
    render(<OutputSection {...baseProps} />);

    const seeDetailBtn = screen.getByText('see detail');
    fireEvent.click(seeDetailBtn);

    const rangeSlider = screen.getByRole('slider');
    const textInput = screen.getByRole('textbox');

    fireEvent.change(rangeSlider, { target: { value: '121000' } });

    expect(rangeSlider.value).toBe('121000');
    expect(textInput.value).toBe('121,000');
  });

  it('shows as disabled when isTraining is true', () => {
    render(<OutputSection {...baseProps} showRetrainBtn={true} isTraining={true} />);

    fireEvent.click(screen.getByText('see detail'));

    const retrainBtn = screen.getByText('Retrain Model');

    expect(retrainBtn).toHaveClass('disabled');
  });

  it('returns null and does not set values if predictData is missing', () => {
    const { container } = render(
      <OutputSection {...baseProps} predictData={null} />
    );

    expect(container.firstChild).toBeNull();

    const input = screen.queryByRole('textbox');
    expect(input).not.toBeInTheDocument();
  });

  it('returns salary value to original prediction when closing details', async () => {
    const user = userEvent.setup();
    render(<OutputSection {...baseProps} />);

    const seeDetailBtn = screen.getByText('see detail');
    const input = screen.getByRole('textbox');

    await user.click(seeDetailBtn);

    await user.clear(input);
    await user.type(input, '150000');
    expect(input.value).toBe('150000');

    await user.click(seeDetailBtn);
    expect(input.value).toBe('120,000');
  });

  it('applies the disabled class to action buttons when isTraining is true', () => {
    render(
      <OutputSection
        {...baseProps}
        isTraining={true}
        showRetrainBtn={true}
      />
    );
    fireEvent.click(screen.getByText('see detail'));

    const addDataBtn = screen.queryByText('Add Data'); 
    const resetBtn = screen.getByText('Training ...'); 
    const retrainBtn = screen.getByText('Retrain Model');

    expect(resetBtn).toHaveClass('disabled');
    expect(retrainBtn).toHaveClass('disabled');

    const input = screen.getByRole('textbox');
    fireEvent.change(input, { target: { value: '150000' } });

    expect(screen.getByText('Add Data')).toHaveClass('disabled');
  });

});
