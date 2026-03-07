import { vi, describe, expect, it, beforeEach, afterEach } from 'vitest';

import { api0 } from '../axiosInstance';
import {
  getUniqJobTitle,
  predictSalary,
  fetchSalaryHistPlot,
  fetchSalaryBoxPlot,
  retrainModel,
  resetModel,
  addData,
  getModelStatus,
} from '../dataService';

let logSpy;

beforeEach(() => {
  vi.resetAllMocks();
  logSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
  globalThis.URL.createObjectURL = vi.fn(() => 'blob:mock');
});

afterEach(() => {
  vi.resetAllMocks();
})

vi.mock('../axiosInstance', () => ({
  api0: {
    get: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
  }
}));

describe('getUniqJobTitle', () => {

  it('return data when success', async () => {
    api0.get.mockResolvedValue({ data: { value: ['Data Scientist'] } });
    const data = await getUniqJobTitle();
    expect(api0.get).toHaveBeenCalledWith('/job_titles');
    expect(data).toStrictEqual({ value: ['Data Scientist'] });
  });

  it('console error message when api error', async () => {
    const err = new Error('test error');
    api0.get.mockRejectedValue(err);
    await getUniqJobTitle();
    expect(logSpy)
    .toHaveBeenCalledWith("Error fetching data:", err.message);
  });
});

describe('predictSalary', () => {
  const formData = {
    age: 26,
    gender: 'male',
    education_level: 'master',
    job_title: 'Data Scientist',
    years_of_experience: 2,
  };

  it('return data when success', async () => {
    api0.post.mockResolvedValue({ data: { value: 1234 } });
    const data = await predictSalary(formData);
    expect(api0.post).toHaveBeenCalledWith('/predictions', formData);
    expect(data).toEqual({ value: 1234 });
  });

  it('console error meddage when catch err', async () => {
    const err = new Error('test error');
    api0.post.mockRejectedValue(err);
    await predictSalary(formData);
    expect(logSpy)
    .toHaveBeenCalledWith("Error predicting salary:", err.message);
  });
});

describe('fetchSalaryHistPlot', () => {
  const salary = 1234;
  const wrongSalary = '';
  const testBlob = new Blob(['test'], { type: 'hist/png' });

  it('return blob URL when success', async () => {
    api0.post.mockResolvedValue({ data: testBlob });
    const url = await fetchSalaryHistPlot(salary);
    expect(api0.post).toHaveBeenCalledWith(
      "/images/histogram",
      { salary },
      { responseType: "blob" },
    );
    expect(globalThis.URL.createObjectURL).toHaveBeenCalledWith(testBlob);
    expect(url).toBe('blob:mock');
  });

  it('return when receiving wrong salary', async () => {
    await fetchSalaryHistPlot(wrongSalary);
    expect(api0.post).not.toHaveBeenCalled();
  });
});

describe('fetchSalaryBoxPlot', () => {
  const salary = 1234;
  const wrongSalary = '';
  const testBlob = new Blob(['test'], { type: 'box/png' });

  it('return blob URL when success', async () => {
    api0.post.mockResolvedValue({ data: testBlob });
    const url = await fetchSalaryBoxPlot(salary);
    expect(api0.post).toHaveBeenCalledWith(
      "/images/boxplot",
      { salary },
      { responseType: "blob" },
    );
    expect(globalThis.URL.createObjectURL).toHaveBeenCalledWith(testBlob);
    expect(url).toBe('blob:mock');
  });

  it('return when receiving wrong salary', async () => {
    await fetchSalaryBoxPlot(wrongSalary);
    expect(api0.post).not.toHaveBeenCalledWith(
      "/salary_boxplot",
      { salary },
      { responseType: "blob" },
    );
  });
});

describe('retrainModel', () => {
  it('calls model retrain api successfully', async () => {
    api0.put.mockResolvedValue({ data: {} });
    await retrainModel();
    expect(api0.put).toHaveBeenCalledWith('/model/training');
  });
});

describe('resetModel', () => {
  it('calls model reset api successfully', async () => {
    api0.put.mockResolvedValue({ data: {} });
    await resetModel();
    expect(api0.put).toHaveBeenCalledWith('/model/initial');
  });
});

describe('addData', () => {
  const formData = {
    age: 39,
    gender: 'male',
    education_level: 'master',
    job_title: 'Data Scientist',
    years_of_experience: 5,
    salary: 130000,
  };

  it('calls add data api when input valid', async () => {
    api0.post.mockResolvedValue({ data: {} });
    await addData(formData);
    expect(api0.post).toHaveBeenCalledWith('/records', formData);
  });
});


describe('getModelStatus', () => {
  it('calls model status api successfully', async () => {
    api0.get.mockResolvedValue({ data: {} });
    await getModelStatus();
    expect(api0.get).toHaveBeenCalledWith('/model/status');
  });
});
