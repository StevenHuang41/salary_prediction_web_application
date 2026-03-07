import { fireEvent, render, screen, waitFor, act } from '@testing-library/react';
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
}));

import {
  getModelStatus,
  predictSalary,
  resetModel,
  retrainModel
} from '../../api/dataService';

describe('HomePage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    getModelStatus.mockResolvedValue({ is_training: false });
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
    predictSalary.mockRejectedValue(
      new Error("predict fails")
    );

    render(<HomePage />);

    fireEvent.click(screen.getByText("Fill Form"));
    fireEvent.click(screen.getByText("Predict Salary"));

    expect(await screen.findByText(/predict fails/)).toBeInTheDocument();
  });

  it("does not call predictSalary when formData is missing", async () => {
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

  it("retrain model success", async () => {
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
    predictSalary.mockResolvedValue({ salary: 100000 });
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

  it("reset model success", async () => {
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

  it("reset model fail", async () => {
    predictSalary.mockResolvedValue({ salary: 100_000 });
    resetModel.mockRejectedValue({});

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

  it('shows model complete retraining toast', async () => {
    getModelStatus
      .mockResolvedValueOnce({ is_training: false })   // mount
      .mockResolvedValueOnce({ is_training: false })  // pollinggetModelStatus.mockResolvedValue({ is_training: true });

    predictSalary.mockResolvedValue({ salary: 100_000 });
    retrainModel.mockResolvedValue({});

    render(<HomePage />);
    fireEvent.click(screen.getByText("Fill Form"));
    fireEvent.click(screen.getByText("Predict Salary"));
    await screen.findByText("OutputSection");

    fireEvent.click(screen.getByText("Retrain")); // click retrain

    await waitFor(() => {
      expect(retrainModel).toHaveBeenCalled();
    });

    await waitFor(() => {
      expect(addToast).toHaveBeenCalledWith(
        "Model training completed!",
        "success"
      );
    }, { timeout: 4000 });

  });

  it('shows model complete reseting toast', async () => {
    getModelStatus
      .mockResolvedValueOnce({ is_training: false })   // mount
      .mockResolvedValueOnce({ is_training: false })  // pollinggetModelStatus.mockResolvedValue({ is_training: true });

    resetModel.mockResolvedValue({});

    render(<HomePage />);
    fireEvent.click(screen.getByText("Fill Form"));
    fireEvent.click(screen.getByText("Predict Salary"));
    await screen.findByText("OutputSection");

    fireEvent.click(screen.getByText("Reset")); // click retrain

    await waitFor(() => {
      expect(resetModel).toHaveBeenCalled();
    });

    await waitFor(() => {
      expect(addToast).toHaveBeenCalledWith(
        "Model resetting completed!",
        "success"
      );
    }, { timeout: 4000 });

  });


  it(
    'set isTraining and trainingType to false and null when pooling fails',
    async () => {
      getModelStatus
        .mockResolvedValueOnce({ is_training: false })   // mount
        .mockRejectedValue(new Error("pooling fails"))  // pollinggetModelStatus.mockResolvedValue({ is_training: true });

      resetModel.mockResolvedValue({});

      render(<HomePage />);
      fireEvent.click(screen.getByText("Fill Form"));
      fireEvent.click(screen.getByText("Predict Salary"));
      await screen.findByText("OutputSection");

      fireEvent.click(screen.getByText("Reset")); // click retrain

      await waitFor(() => {
        expect(getModelStatus).toHaveBeenCalled();
      });

      await waitFor(() => {
        expect(getModelStatus).toHaveBeenCalledTimes(2);
      }, { timeout: 4000 });

    }
  );

});
