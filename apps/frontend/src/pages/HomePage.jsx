import { useState, useEffect } from 'react';
import {
  predictSalary,
  retrainModel,
  resetModel,
  getModleStatus,
} from '../api/dataService';

import InputForm from '../components/InputForm';
import OutputSection from '../components/OutputSection';
import ErrorPredict from '../components/ErrorPredict';
import LoadingResult from '../components/LoadingResult';
import useToast from '../hooks/useToast'
import MyToast from '../components/Toast';

const HomePage = () => {
  const [predictState, setPredictState] = useState({
    data: null,
    loading: false,
    error: null,
  });

  const [formData, setFormData] = useState(null);
  const [showDetail, setShowDetail] = useState(false);

  const [isTraning, setIsTraning] = useState(false);
  
  const { toasts, addToast, removeToast } = useToast();

  const handleInputFormSubmit = async () => {
    setLoadingResult(true);
    setErrResult(null);

    try {
      // console.log(formData);
      const res = await predictSalary(formData);
      // console.log(res);
      setPredictResult(res);
    } catch (err) {
      setErrResult(err.message);
    } finally {
      setLoadingResult(false)
    }
  };

  return (<>
    <div className="container">
      <InputForm
        getSubmitData={setFormData}
        handleInputFormSubmit={handleInputFormSubmit}
        setPredictResult={setPredictResult}
        showDetail={showDetail}
        addToast={addToast}
        loadingFunc={loadingResult}
        setLoadingFunc={setLoadingResult}
        setErrFunc={setErrResult}
        dataAdded={dataAdded}
        setDataAdded={setDataAdded}
      />

      {errResult ?
      <ErrorPredict data={errResult}/>
      :
      loadingResult ?
      <div className="loading-container">
        <LoadingResult
          loadingText="Loading ..."
          setStyle={{fontSize: "5em"}}
          setClass="mt-5 mt-sm-3"
          setTextClass="d-none d-sm-flex"
        />
      </div>
      :
      predictResult &&
      <OutputSection
        dataFromForm={formData}
        predictData={predictResult}
        setErrFunc={setErrResult}
        addToast={addToast}
        showDetail={showDetail}
        setShowDetail={setShowDetail}
        setDataAdded={setDataAdded}
      />}

      {/* toasts */}
      <MyToast
        toasts={toasts}
        removeToast={removeToast}
      />

    </div>
  </>)
};


export default HomePage;
