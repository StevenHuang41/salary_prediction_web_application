import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { vi, describe, expect, it, beforeEach } from 'vitest';
import HomePage from '../HomePage';

vi.mock('../../components/InputForm', () => ({
  default: ({ onSubmit, setFormData }) => (
    <div>
      <button
        onClick={() =>
          setFormData({
            age: 27,
            gender: 'male',
            education_level: 'master',
            job_title: 'Data Scientist',
            years_of_experience: 0,
          })
        }
      >
        Fill Form
      </button>

      <button onClick={onSubmit}>
        Predict Salary
      </button>
    </div>
  )
}));

vi.mock('../../components/OutputSection', () => ({
  default: ({ onRetrain, onReset }) => (
    <div>
      <div>OutputSection</div>

      <button onClick={onRetrain}>
        Retrain
      </button>

      <button onClick={onReset}>
        Reset
      </button>
    </div>
  )
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

const addToast = vi.fn();
vi.mock('../../hooks/useToast', () => ({
  default: () => ({
    toasts: [],
    addToast,
    removeToast: vi.fn(),
  }),
}));

vi.mock('../../api/dataService', () => ({
  predictSalary: vi.fn(),
  retrainModel: vi.fn(),
  resetModel: vi.fn(),
  getModelStatus: vi.fn(),
  modelDataSync: vi.fn(),
}));

import {
  predictSalary,
  resetModel,
  retrainModel,
  getModelStatus,
  modelDataSync,
} from '../../api/dataService';

describe('HomePage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders HomePage components', () => {
    render(<HomePage />)

    expect(screen.getByText("Fill Form")).toBeInTheDocument();
    expect(screen.getByText('Predict Salary')).toBeInTheDocument();
    expect(screen.getByTestId('MyToast')).toBeInTheDocument();
  });

  it('shows loading and then output section', async () => {
    predictSalary.mockImplementation(() =>
      new Promise(resolve =>
        setTimeout(() => resolve({ salary: 100_000 }), 50)
      )
    );

    render(<HomePage />);

    fireEvent.click(screen.getByText('Fill Form'));
    fireEvent.click(screen.getByText('Predict Salary'));

    expect(await screen.findByText(/Loading .../i)).toBeInTheDocument();
    expect(await screen.findByText('OutputSection')).toBeInTheDocument();
  });

  it("renders error component when prediction fails", async () => {
    predictSalary.mockRejectedValue(new Error("predict fails"));

    render(<HomePage />);

    fireEvent.click(screen.getByText("Fill Form"));
    fireEvent.click(screen.getByText("Predict Salary"));

    expect(await screen.findByText(/predict fails/)).toBeInTheDocument();
  });

  it("does not call predictSalary when missing formData", async () => {
    render(<HomePage />);

    fireEvent.click(screen.getByText("Predict Salary"));

    expect(predictSalary).not.toHaveBeenCalled();
  });

  it("checks model status on mount", async () => {
    render(<HomePage />);

    expect(getModelStatus).toHaveBeenCalled();
  });

  it("logs error when getModelStatus fails on mount", async () => {
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    getModelStatus.mockRejectedValue(new Error('api failed'));

    render(<HomePage />);

    await waitFor(() => {
      expect(consoleSpy).toHaveBeenCalledWith(expect.any(Error));
    });

    consoleSpy.mockRestore()
  });

  // retrain
  it("retrain success shows in toast", async () => {
    predictSalary.mockResolvedValue({ salary: 100_000 });
    retrainModel.mockResolvedValue({});

    render(<HomePage />);

    fireEvent.click(screen.getByText("Fill Form"));
    fireEvent.click(screen.getByText("Predict Salary"));

    await screen.findByText("OutputSection");

    fireEvent.click(screen.getByText("Retrain"));

    await waitFor(() => {
      expect(retrainModel).toHaveBeenCalled();
    });

    expect(addToast).toHaveBeenCalledWith(
      "Model retraining ...",
      "info"
    );
  });

  it("retrain failure shows toast", async () => {
    predictSalary.mockResolvedValue({ salary: 100_000 });
    retrainModel.mockRejectedValue(new Error());

    render(<HomePage />);

    fireEvent.click(screen.getByText("Fill Form"));
    fireEvent.click(screen.getByText("Predict Salary"));

    await screen.findByText("OutputSection");

    fireEvent.click(screen.getByText("Retrain"));

    await waitFor(() => {
      expect(addToast).toHaveBeenCalledWith(
        "Failed to retrain model!",
        "danger"
      );
    });
  });

  //reset
  it("reset success shows in toast", async () => {
    predictSalary.mockResolvedValue({ salary: 100_000 });
    resetModel.mockResolvedValue({});

    render(<HomePage />);

    fireEvent.click(screen.getByText("Fill Form"));
    fireEvent.click(screen.getByText("Predict Salary"));

    await screen.findByText("OutputSection");

    fireEvent.click(screen.getByText("Reset"));

    await waitFor(() => {
      expect(resetModel).toHaveBeenCalled();
    });

    expect(addToast).toHaveBeenCalledWith(
      "Model resetting ...",
      "info"
    );
  });

  it("reset failure shows in toast", async () => {
    predictSalary.mockResolvedValue({ salary: 100_000 });
    resetModel.mockRejectedValue(new Error());

    render(<HomePage />);

    fireEvent.click(screen.getByText("Fill Form"));
    fireEvent.click(screen.getByText("Predict Salary"));

    await screen.findByText("OutputSection");

    fireEvent.click(screen.getByText("Reset"));

    await waitFor(() => {
      expect(resetModel).toHaveBeenCalled();
    });

    expect(addToast).toHaveBeenCalledWith(
      "Failed to reset model!",
      "danger"
    );
  });

  it("shows toast when is still training", async () => {
    getModelStatus.mockResolvedValue({ is_training: true });

    render(<HomePage />);

    await waitFor(() => {
      expect(addToast).toHaveBeenCalledTimes(1);
    });

    expect(addToast).toHaveBeenCalledWith(
      "Model is still training ...",
      "info"
    )
  });

  it('shows model completed retraining toast', async () => {
    predictSalary.mockResolvedValue({ salary: 100_000 });

    getModelStatus
      .mockResolvedValueOnce({ is_training: false })  // mount
      .mockResolvedValueOnce({ is_training: true })
      .mockResolvedValueOnce({ is_training: false })

    retrainModel.mockResolvedValue({});

    render(<HomePage />);

    fireEvent.click(screen.getByText("Fill Form"));
    fireEvent.click(screen.getByText("Predict Salary"));

    await screen.findByText("OutputSection");

    vi.useFakeTimers();
    fireEvent.click(screen.getByText("Retrain"));

    expect(addToast).toHaveBeenCalledWith("Model retraining ...", "info");

    await vi.advanceTimersByTimeAsync(5000);
    expect(addToast).not.toHaveBeenCalledWith(
      "Model training completed!",
      "success"
    );

    await vi.advanceTimersByTimeAsync(5000);
    expect(addToast).toHaveBeenCalledWith(
      "Model training completed!",
      "success"
    );
    vi.useRealTimers();
    await expect(modelDataSync).toHaveBeenCalled();
  });

  it('shows model complete reseting toast', async () => {
    predictSalary.mockResolvedValue({ salary: 100_000 });

    getModelStatus
      .mockResolvedValueOnce({ is_training: false })  // mount
      .mockResolvedValueOnce({ is_training: true })
      .mockResolvedValueOnce({ is_training: false })

    resetModel.mockResolvedValue({});

    render(<HomePage />);

    fireEvent.click(screen.getByText("Fill Form"));
    fireEvent.click(screen.getByText("Predict Salary"));

    await screen.findByText("OutputSection");

    vi.useFakeTimers()
    fireEvent.click(screen.getByText("Reset"));

    expect(addToast).toHaveBeenCalledWith(
      "Model resetting ...",
      "info"
    );

    await vi.advanceTimersByTimeAsync(5000);
    expect(addToast).not.toHaveBeenCalledWith(
      "Model resetting completed!",
      "success"
    );

    await vi.advanceTimersByTimeAsync(5000);
    expect(addToast).toHaveBeenCalledWith(
      "Model resetting completed!",
      "success"
    );

    vi.useRealTimers();
  });


  it(
    'set isTraining and trainingType to false and null when polling fails',
    async () => {
      getModelStatus
        .mockResolvedValueOnce({ is_training: false })   // mount
        .mockRejectedValue(new Error("pooling fails"))  // pollinggetModelStatus.mockResolvedValue({ is_training: true });

      resetModel.mockResolvedValue({});

      render(<HomePage />);

      fireEvent.click(screen.getByText("Fill Form"));
      fireEvent.click(screen.getByText("Predict Salary"));

      await screen.findByText("OutputSection");

      vi.useFakeTimers()
      fireEvent.click(screen.getByText("Reset"));

      expect(getModelStatus).toHaveBeenCalled();

      expect(addToast).toHaveBeenCalledWith(
        "Model resetting ...",
        "info"
      );

      await vi.advanceTimersByTimeAsync(5000);
      addToast();
      expect(addToast).toHaveBeenCalledWith(
        "Failed to get model status",
        "danger"
      );
      vi.useRealTimers();
    }
  );

});
