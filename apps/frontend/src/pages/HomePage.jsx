import { useState, useEffect, useRef } from 'react';
import {
  predictSalary,
  retrainModel,
  resetModel,
  getModelStatus,
  modelDataSync,
} from '../api/dataService';

import InputForm from '../components/InputForm';
import OutputSection from '../components/OutputSection';
import ErrorPredict from '../components/ErrorPredict';
import LoadingResult from '../components/LoadingResult';
import useToast from '../hooks/useToast';
import MyToast from '../components/Toast';

const HomePage = () => {
  const [predictState, setPredictState] = useState({
    data: null,
    loading: false,
    error: null,
  });

  const [formData, setFormData] = useState(null);

  const [isTraining, setIsTraining] = useState(false);
  const [trainingType, setTrainingType] = useState(null);

  const { toasts, addToast, removeToast } = useToast();

  const toastShownRef = useRef(false);
  const hasShownReloadToast = useRef(false);

  const [dbChanged, setDBChanged] = useState(false);

  const showRetrainBtn = dbChanged && !isTraining

  useEffect(() => {
    const checkStatus = async () => {
      try {
        const res = await getModelStatus();
        setIsTraining(res.is_training);
        if (res.is_training && !hasShownReloadToast.current) {
          hasShownReloadToast.current = true;
          addToast("Model is still training ...", "info");
        }
      } catch (err) {
        console.error(err);
      }
    };

    checkStatus();
  }, []);

  const handlePredict = async () => {
    if (!formData) return;

    setPredictState({
      data: null,
      loading: true,
      error: null,
    });

    try {
      const res = await predictSalary(formData);

      setPredictState({
        data: res,
        loading: false,
        error: null,
      });

    } catch (err) {
      setPredictState({
        data: null,
        loading: false,
        error: err.message,
      });
    }

  };

  const handleRetrain = async () => {
    try {
      setIsTraining(true);
      setTrainingType("retrain")
      addToast("Model retraining ...", "info");
      await retrainModel();
      setDBChanged(false);
    } catch (err) {
      setIsTraining(false);
      setTrainingType(null)
      addToast("Failed to retrain model!", "danger")
    }
  };

  const handleReset = async () => {
    try {
      setIsTraining(true);
      setTrainingType("reset")
      addToast("Model resetting ...", "info");
      await resetModel();
      setDBChanged(true);
    } catch (err) {
      setIsTraining(false);
      setTrainingType(null)
      addToast("Failed to reset model!", "danger")
    }
  };

  useEffect(() => {
    if (!isTraining) return;

    toastShownRef.current = false;

    const interval = setInterval(async () => {
      try {
        const res = await getModelStatus();

        if (!res.is_training && !toastShownRef.current) {
          toastShownRef.current = true

          if (trainingType === "retrain") {
            addToast("Model training completed!", "success");
            modelDataSync();
          }
          if (trainingType === "reset") {
            addToast("Model resetting completed!", "success");
          }

          setIsTraining(false);
          setTrainingType(null)
        }
      } catch (err) {
        addToast("Failed to get model status", "danger");
        setIsTraining(false);
        setTrainingType(null)
      }
    }, 5000)
    return () => clearInterval(interval);
  }, [isTraining]);

  const renderPredictSection = () => {
    if (predictState.error) {
      return <ErrorPredict data={predictState.error} />;
    }

    if (predictState.loading) {
      return (
        <div className="loading-container">
          <LoadingResult
            loadingText="Loading ..."
            setStyle={{fontSize: "5em"}}
            setClass="mt-5 mt-sm-3"
            setTextClass="d-none d-sm-flex"
          />
        </div>
      );
    }

    if (predictState.data) {
      return (
        <OutputSection
          dataFromForm={formData}
          predictData={predictState.data}
          addToast={addToast}
          isTraining={isTraining}
          onRetrain={handleRetrain}
          onReset={handleReset}
          setDBChanged={setDBChanged}
          showRetrainBtn={showRetrainBtn}
        />
      )
    }

    return null;
  };

  return (<>
    <div className="container mb-5">
      <InputForm
        onSubmit={handlePredict}
        setPredictState={setPredictState}
        setFormData={setFormData}
        isPredicting={predictState.loading}
      />

      {renderPredictSection()}

      {/* toasts */}
      <MyToast
        toasts={toasts}
        removeToast={removeToast}
      />

    </div>
  </>)
};

export default HomePage;
