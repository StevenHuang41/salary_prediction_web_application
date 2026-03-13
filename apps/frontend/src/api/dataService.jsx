import { api0 } from "./axiosInstance";


const getUniqJobTitle = async () => {
  try {
    const res = await api0.get('/job_titles');
    return res.data;
  } catch (err) {
    console.error("Error fetching data:", err.message);
  }
};

const predictSalary = async (formData) => {
  try {
    const res = await api0.post('/predictions', formData);
    return res.data;
  } catch (err) {
    console.error("Error predicting salary:", err.message);
  }
};

const fetchSalaryHistPlot = async (salary) => {
  if (salary === '') return ;

  const res = await api0.post(
    "/images/histogram",
    { salary },
    { responseType: "blob" },
  );

  return URL.createObjectURL(res.data);
};

const fetchSalaryBoxPlot = async (salary) => {
  if (salary === '') return ;

  const res = await api0.post(
    "/images/boxplot",
    { salary },
    { responseType: "blob" },
  );

  return URL.createObjectURL(res.data);
};

const addData = async (data) => {
  const res = await api0.post('/records', data);
  return res.data;
}

const retrainModel = async () => {
  const res = await api0.put('/model/training');
  return res.data;
};

const resetModel = async () => {
  const res = await api0.put('/model/initial');
  return res.data;
};

const getModelStatus = async () => {
  try {
    const res = await api0.get('/model/status');
    return res.data;
  } catch (err) {
    console.error(err)
    return { is_training: false };
  }
}

const modelDataSync = async () => {
  const res = await api0.put('/model/data-sync');
  return res.data
}


export {
  getUniqJobTitle,
  predictSalary,
  fetchSalaryHistPlot,
  fetchSalaryBoxPlot,
  retrainModel,
  resetModel,
  addData,
  getModelStatus,
  modelDataSync,
};
