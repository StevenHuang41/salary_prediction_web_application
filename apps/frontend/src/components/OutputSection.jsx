import { useEffect, useState } from "react";
import "./OutputSection.css";
import {
  fetchSalaryBoxPlot,
  fetchSalaryHistPlot,
  addData,
} from "../api/dataService";

import MyCarousel from "./MyCarousel";
import LoadingResult from "./LoadingResult";

const OutputSection = ({
  dataFromForm,
  predictData,
  addToast,
  isTraining,
  onRetrain,
  onReset,
  setDBChanged,
  showRetrainBtn,
}) => {
  const [salaryInput, setSalaryInput] = useState("");

  const [rangeValue, setRangeValue] = useState(0);

  const [showDetail, setShowDetail] = useState(false);

  const [imgURLs, setImgURLs] = useState({});
  const imgFetchers = {
    hist: fetchSalaryHistPlot,
    box: fetchSalaryBoxPlot,
  }

  const [isValid, setIsValid] = useState(false);

  const [isSameAsPredict, setIsSameAsPredict] = useState(true);

  // show predict salary, updates when predictData changes
  useEffect(() => {
    if (!predictData) return;

    const salary = Number(predictData.salary);

    // set ',' in salary string
    setSalaryInput(
      salary.toLocaleString("en-US", {
        maximumFractionDigits: 2,
      }),
    );

    setRangeValue(salary);
  }, [predictData]);

  const parse2Number = (value) => {
    if (!value) return NaN;
    return Number(value.replace(/,/g, ""));
  };

  // updates when salaryInput changes
  useEffect(() => {
    if (!predictData) return;

    const parsed = parse2Number(salaryInput);

    // if input is invalid, return
    if (isNaN(parsed)) {
      setIsValid(false);
      setIsSameAsPredict(false);
      return;
    }

    const predicted = Number(predictData.salary).toFixed(2);
    const changed = parsed.toFixed(2);

    setIsSameAsPredict(predicted === changed);
    // changed value valid and not the same as original
    setIsValid(predicted !== changed);

    const fetchPlots = async () => {

      Object.entries(imgFetchers).forEach(([key, fetcher]) => {
        fetcher(parsed)
          .then((img) => {
            setImgURLs(prev => ({ ...prev, [key]: img }));
          })
          .catch((err) => {
            console.error(err);
          });
      });

    };
    const timeout = setTimeout(fetchPlots, 200);
    return () => clearTimeout(timeout);
  }, [salaryInput, predictData]);

  if (!predictData) return null; //////////////////////////////////////////

  // handle input of predict salary change
  const handleSalaryChange = (value) => {
    setSalaryInput(value);
    setRangeValue(parse2Number(value));
  };

  // handle range input of predict salary change
  const handleRangeChange = (value) => {
    const numeric = parse2Number(value)
    setRangeValue(numeric);
    setSalaryInput(
      numeric.toLocaleString("en-US", {
        maximumFractionDigits: 2,
      }),
    );
  };

  // handle return btn click
  const handleReturn = () => {
    const original = Number(predictData.salary);
    setRangeValue(original);
    setSalaryInput(
      original.toLocaleString("en-US", {
        maximumFractionDigits: 2,
      }),
    );
  };

  // handel add data btn click
  const handleAddData = async () => {
    const new_record = {
      ...dataFromForm,
      salary: parse2Number(salaryInput),
    }
    try {
      await addData(new_record);
      setDBChanged(true);
      addToast("Data added successfully!", "success");
    } catch (err) {
      addToast("Failed to add data", "danger");
    }
  };

  const renderCarousel = () => {
    if (!imgURLs.hist) {
      return (
        <LoadingResult
          loadingText="Loading carousel images"
          setStyle={{ fontSize: "2em", height: "15vh" }}
        />
      )
    }

    return (
      <div className="row mx-0">
        <div className="col d-flex justify-content-center px-0">
          <MyCarousel
            images={Object.values(imgURLs)}
            alts={["Salary Histogram Plot", "Salary Box Plot"]}
          />
        </div>
      </div>
    )
  }

  return (<>
    {/* predict salary value */}
    <div
      className={`
      row
      mx-0 my-2
      d-flex
      justify-content-center
      align-items-center
      `}
    >
      <input
        id="predict-input"
        className={`
          col-12
          form-control
          fw-bold text-center w-100
        `}
        value={salaryInput}
        onChange={(e) => handleSalaryChange(e.target.value)}
      />
    </div>

    {/*salary input range */}
    {showDetail && (
    <div className="row">
      <div className="col">
        <input
          type="range"
          className="form-range"
          min={(predictData.salary - predictData.mae).toFixed(2)}
          max={(predictData.salary + predictData.mae).toFixed(2)}
          step="0.01"
          value={rangeValue}
          onChange={(e) => handleRangeChange(e.target.value)}
        />
      </div>
    </div>
    )}

    {/* see detial row */}
    <div
      className={`
        row
        mx-0 gap-1
        d-flex
        align-items-center
      `}
    >
      <div className="col order-2 order-md-1 px-0">
        {/* {isSameAsOriginal && !isValid && ( */}
          <div className="row">
            <div className="col-12">
              Model {showDetail && `Name`}: {predictData.model_name}
              {/* <br /> */}
            </div>
            <div className="col-12">
              {showDetail ? `Mean Absolute Error` : `MAE`}:{" "}
              {predictData.mae.toFixed(2)}
            </div>
          </div>
        {/* )} */}
      </div>

      {/* btn see detail */}
      <div
        className={`
        col-12 col-md-auto
        px-0
        d-flex
        justify-content-md-end
        order-1 order-md-2
        `}
      >
        <div
          className={`
          btn
          p-2 py-1
          text-nowrap
          ${showDetail ? `btn-secondary` : `btn-outline-secondary`}
          `}
          onClick={() => {
            if (showDetail) handleReturn();
            setShowDetail(!showDetail);
          }}
        >
          see detail
        </div>
      </div>
    </div>

    {/* Carousel */}
    {!showDetail && renderCarousel()}

    {/* detail of model */}
    {showDetail && (<>
      <div className={`row row-cols-1 mb-2`}>
          <div className="col">
            Mean Square Error: {predictData.mse.toFixed(2)}
          </div>
          <div className="col">
            Root Mean Square Error: {predictData.rmse.toFixed(2)}
          </div>
          <div className="col">
            Train size: {predictData.n_train}
          </div>
          <div className="col">
            Test size: {predictData.n_test}
          </div>
          <div className="col">
            Created At: {predictData.created_at}
          </div>
          <div className="col">
            Duration: {predictData.duration}
          </div>
      </div>

      <div className="row g-2">
          {showDetail && !isSameAsPredict && (<>
          <div className="col-auto order-2 order-md-1">
            <div
              className={`
                btn btn-outline-success
                p-2 py-1
                text-nowrap
              `}
              onClick={handleReturn}
            >
              Return Input
            </div>
          </div>

          {isValid && (
          <div className="col-auto order-2 order-md-1">
            <div
              className={`
                btn btn-outline-info
                p-2 py-1
                text-nowrap
                ${isTraining && `disabled`}
              `}
              onClick={handleAddData}
            >
              Add Data
            </div>
          </div>
          )}
          </>)}

          <div className="col-auto order-2 order-md-1">
            <div
              className={`
                btn btn-outline-danger
                p-2 py-1
                text-nowrap
                ${isTraining && `disabled`}
              `}
              onClick={onReset}
            >
              {!isTraining ? `Reset Database` : `Training ...`}
            </div>
          </div>

          {showRetrainBtn && (
          <div className="col-auto order-2 order-md-1">
            <div
              className={`
                btn btn-outline-warning
                p-2 py-1
                text-nowrap
                ${isTraining && `disabled`}
              `}
              onClick={onRetrain}
            >
              Retrain Model
            </div>
          </div>
          )}
      </div>
    </>)}

    {showDetail && (
    <div className="row row-cols-1 mt-3 px-0">
      <img
        className={`
        col
        img-fluid
        mb-2
        `}
        src={imgURLs.hist}
        alt="Salary Histogram Plot"
      />

      <img
        className={`
        col
        img-fluid
        `}
        src={imgURLs.box}
        alt="Salary Box Plot"
      />
    </div>
    )}
  </>);
};

export default OutputSection;
