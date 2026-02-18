import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { vi, describe, expect, it, beforeEach } from 'vitest';
import OutputSection from '../OutputSection';
import userEvent from '@testing-library/user-event';

vi.mock('../../api/dataService', () => ({
  fetchSalaryHistPlot: vi.fn(),
  fetchSalaryBoxPlot: vi.fn(),
  resetModel: vi.fn(),
  addData: vi.fn(),
}));

import {
  fetchSalaryHistPlot,
  fetchSalaryBoxPlot,
  resetModel,
  addData,
} from '../../api/dataService';

vi.mock('../MyCarousel', () => ({
  default: () => (
    <div data-testid="MyCarousel">
      <img src={null} alt='carouselImg1' />
      <img src={null} alt='carouselImg2' />
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
    value: 120000,
    model_name: 'randomForestRegressor',
    use_polynomial: true,
    params: {
      mae: 3000,
      mse: 8000000,
    },
    num_train_dataset: 4000,
    num_test_dataset: 1000
  },
  setErrFunc: vi.fn(),
  addToast: vi.fn(),
  showDetail: false,
  setShowDetail: vi.fn(),
  setDataAdded: vi.fn(),
};

beforeEach(() => {
  vi.clearAllMocks();
});


const clearInputAndExpectEmpty = async (input) => {
  await userEvent.clear(input);
  await waitFor(() => expect(input.value).toBe(''));
};

describe('OutputSection', () => {
  it('render output components when see detail is false', async () => {
    render(<OutputSection {...baseProps} />);
    expect(document.querySelector('input#predict-input')).toBeInTheDocument();
    expect(screen.getByText(/Model/)).toBeInTheDocument();
    expect(screen.getByText(/MAE/)).toBeInTheDocument();

    const seeDetailBtn = screen.getByText('see detail');
    expect(seeDetailBtn).toBeInTheDocument();
    expect(seeDetailBtn).toHaveClass('btn-outline-secondary');

    expect(await screen.findByTestId('MyCarousel')).toBeInTheDocument();
    expect(await screen.findByAltText('carouselImg1')).toBeInTheDocument();
    expect(await screen.findByAltText('carouselImg2')).toBeInTheDocument();

    // should not show when see detail is false
    expect(screen.queryByRole('slider')).not.toBeInTheDocument();
    expect(seeDetailBtn).not.toHaveClass('btn-secondary');
    expect(screen.queryByText(/Model Name/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Mean Absolute Error/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Mean Square Error/)).not.toBeInTheDocument();
    expect(screen.queryByText(/#Train dataset/)).not.toBeInTheDocument();
    expect(screen.queryByText(/#Test dataset/)).not.toBeInTheDocument();
    expect(screen.queryByText('Reset Database')).not.toBeInTheDocument();
    expect(screen.queryByAltText('Salary Histogram Plot')).not.toBeInTheDocument();
    expect(screen.queryByAltText('Salary Box Plot')).not.toBeInTheDocument();
  });

  it('render output components when see detail is true', async () => {
    render(<OutputSection {...baseProps} showDetail={true} />);
    expect(screen.getByRole('slider')).toBeInTheDocument();

    const seeDetailBtn = screen.getByText('see detail');
    expect(seeDetailBtn).toBeInTheDocument();
    expect(seeDetailBtn).toHaveClass('btn-secondary');

    expect(screen.getByText(/Model Name/)).toBeInTheDocument();
    expect(screen.getByText(/Mean Absolute Error/)).toBeInTheDocument();
    expect(screen.getByText(/Mean Square Error/)).toBeInTheDocument();
    expect(screen.getByText(/#Train dataset/)).toBeInTheDocument();
    expect(screen.getByText(/#Test dataset/)).toBeInTheDocument();
    expect(screen.getByText('Reset Database')).toBeInTheDocument();
    expect(await screen.findByAltText('Salary Histogram Plot')).toBeInTheDocument();
    expect(await screen.findByAltText('Salary Box Plot')).toBeInTheDocument();
  });

  it('does not update predict input when predictData is null', () => {
    render(<OutputSection {...baseProps} predictData={null}/>)
    expect(document.querySelector('input#predict-input')).not.toBeInTheDocument();
  });


  it(
    'render output components when see detail is true and change predict input',
    async () => {
      render(<OutputSection {...baseProps} showDetail={true} />);

      const predictInput = document.querySelector('input#predict-input');

      // clear input 
      clearInputAndExpectEmpty(predictInput);
      // input is not a valid number ('')
      expect(screen.getByText('Return Input')).toBeInTheDocument();
      expect(screen.queryByText('Add Data')).not.toBeInTheDocument();

      await userEvent.type(predictInput, '123');
      // input is a valid number ('123')
      expect(screen.getByText('Return Input')).toBeInTheDocument();
      expect(screen.getByText('Add Data')).toBeInTheDocument();
    }
  );


  it('update predictSalary when input change via textbox', async () => {
    render(<OutputSection {...baseProps} />);
    const predictInput = document.querySelector('input#predict-input');

    // clear textbox
    clearInputAndExpectEmpty(predictInput);

    await userEvent.type(predictInput, '12345');
    expect(predictInput.value).toBe('12345');
  });

  it('update predictSalary and rangeValue via textbox (showDetail)', async () => {
    render(<OutputSection {...baseProps} showDetail={true}/>);
    const predictInput = document.querySelector('input#predict-input');

    // clear textbox
    clearInputAndExpectEmpty(predictInput);

    await userEvent.type(predictInput, '119000');
    expect(predictInput.value).toBe('119000');
    expect(screen.getByRole('slider').value).toBe('119000');
    
    // clear textbox
    clearInputAndExpectEmpty(predictInput);

    await userEvent.type(predictInput, '12345');
    expect(predictInput.value).toBe('12345');
    expect(screen.getByRole('slider').value).toBe('117000');

    // clear textbox
    clearInputAndExpectEmpty(predictInput);

    await userEvent.type(predictInput, '999999');
    expect(predictInput.value).toBe('999999');
    expect(screen.getByRole('slider').value).toBe('123000');
  });

  it('change predict input value (en-US) when toggle rangeBar', async () => {
    render(<OutputSection {...baseProps} showDetail={true}/>);

    const predictInput = document.querySelector('input#predict-input');
    const rangeBar = screen.getByRole('slider');

    fireEvent.change(rangeBar, { target: { value: '121000' } });
    expect(rangeBar).toHaveValue('121000');
    expect(predictInput).toHaveValue('121,000');
  });

  it('undo the value of predict input by clicking Return Input btn', async () => {
    render(<OutputSection {...baseProps} showDetail={true}/>);
    const predictInput = document.querySelector('input#predict-input');
    
    // change input to ''
    clearInputAndExpectEmpty(predictInput);

    const returnInputBtn = screen.getByText('Return Input');
    expect(returnInputBtn).toBeInTheDocument();

    await userEvent.click(returnInputBtn);
    expect(predictInput.value).toBe('120,000');

    expect(screen.queryByText('Return Input')).not.toBeInTheDocument();
  })

  it('undo the predict input when closing see detail btn', async () => {
    render(<OutputSection {...baseProps} showDetail={true}/>);
    const predictInput = document.querySelector('input#predict-input');
    const seeDetailBtn = screen.getByText('see detail');

    // change input to ''
    clearInputAndExpectEmpty(predictInput);

    await userEvent.click(seeDetailBtn);
    await waitFor(() => {
      expect(predictInput.value).toBe('120,000');
    })
  });

  it('reset database by clicking Reset Database btn (resolve)', async () => {
    resetModel.mockResolvedValue({
      status: 'sucess',
      message: 'Reset Database successfully.',
    })
    render(<OutputSection {...baseProps} showDetail={true}/>);
    const resetDatabaseBtn = screen.getByText('Reset Database');
    await userEvent.click(resetDatabaseBtn);
    await waitFor(() => {
      expect(baseProps.setErrFunc).toHaveBeenCalled();
      expect(baseProps.addToast).toHaveBeenCalled(1);

      expect(baseProps.addToast)
      .toHaveBeenCalledWith("Reset database ...", "secondary");

      expect(resetModel).toHaveBeenCalled();
      expect(baseProps.setDataAdded).toHaveBeenCalledWith(true);
      expect(baseProps.addToast).toHaveBeenCalled(2);

      expect(baseProps.addToast)
      .toHaveBeenCalledWith("Reset database successfully", "success");
    })
  });

  it('reset database by clicking Reset Database btn (reject)', async () => {
    resetModel.mockRejectedValue({
      status: 'fail',
      message: 'Reset Database failed.',
    });
    render(<OutputSection {...baseProps} showDetail={true}/>);
    const resetDatabaseBtn = screen.getByText('Reset Database');
    await userEvent.click(resetDatabaseBtn);
    await waitFor(() => {
      expect(baseProps.setErrFunc).toHaveBeenCalled(null);
      expect(baseProps.addToast).toHaveBeenCalled(1);

      expect(baseProps.addToast)
      .toHaveBeenCalledWith("Reset database ...", "secondary");

      expect(resetModel).toHaveBeenCalled();
      expect(baseProps.setDataAdded).not.toHaveBeenCalledWith(true);
      expect(baseProps.addToast).toHaveBeenCalled(1);

      expect(baseProps.setErrFunc).toHaveBeenCalled(2);
      expect(baseProps.addToast)
      .toHaveBeenCalledWith("Reset database failed", "danger");
    });
  });

  it('add data to database by clicking Add Data btn (resolve)', async () => {
    addData.mockResolvedValue({
      data: {
        status: 'sucess',
        message: 'Add data to database.'
      }
    });
    render(<OutputSection {...baseProps} showDetail={true}/>);
    const predictInput = document.querySelector('input#predict-input');
    // // change input to ''
    clearInputAndExpectEmpty(predictInput);
    await userEvent.type(predictInput, '123,456');

    const addDataBtn = screen.getByText('Add Data');
    await userEvent.click(addDataBtn);
    await waitFor(() => {
      expect(baseProps.setErrFunc).toHaveBeenCalled(null);
      expect(addData).toHaveBeenCalled();
      expect(baseProps.setDataAdded).toHaveBeenCalled();
      expect(baseProps.addToast)
      .toHaveBeenCalledWith("Data added successfully!", "success");
    })
  });

  it('add data to database by clicking Add Data btn (reject)', async () => {
    addData.mockRejectedValue({
      data: {
        status: 'fail',
        message: 'Data does not add to Database.'
      }
    });
    render(<OutputSection {...baseProps} showDetail={true}/>);
    const predictInput = document.querySelector('input#predict-input');
    // // change input to ''
    clearInputAndExpectEmpty(predictInput);
    await userEvent.type(predictInput, '123,456');

    const addDataBtn = screen.getByText('Add Data');
    await userEvent.click(addDataBtn);
    await waitFor(() => {
      expect(baseProps.setErrFunc).toHaveBeenCalledWith(null);
      expect(addData).toHaveBeenCalled();
      expect(baseProps.setDataAdded).not.toHaveBeenCalled();
      expect(baseProps.addToast)
      .not.toHaveBeenCalledWith("Data added successfully!", "success");
      expect(baseProps.setErrFunc).toHaveBeenCalled(2);
    })
  });

  it('catch error when fetchSalaryHistPlot fail', async () => {
    fetchSalaryHistPlot.mockRejectedValue(new Error('hist plot error'));
    fetchSalaryBoxPlot.mockResolvedValue('test-url');
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {});

    render(<OutputSection {...baseProps} showDetail={true}/>);
    const predictInput = document.querySelector('input#predict-input');
    
    clearInputAndExpectEmpty(predictInput);
    await userEvent.type(predictInput, '123456');

    await waitFor(() => {
      expect(logSpy).toHaveBeenCalledWith(expect.any(Error));
      expect(logSpy.mock.calls[0][0].message).toBe('hist plot error');
    });

    logSpy.mockRestore();
  });

  it('catch error when fetchSalaryBoxPlot fail', async () => {
    fetchSalaryHistPlot.mockResolvedValue('test-url');
    fetchSalaryBoxPlot.mockRejectedValue(new Error('box plot error'));
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {});

    render(<OutputSection {...baseProps} showDetail={true}/>);
    const predictInput = document.querySelector('input#predict-input');
    
    clearInputAndExpectEmpty(predictInput);
    await userEvent.type(predictInput, '123456');

    await waitFor(() => {
      expect(logSpy).toHaveBeenCalledWith(expect.any(Error));
      expect(logSpy.mock.calls[0][0].message).toBe('box plot error');
    });

    logSpy.mockRestore();
  });
});