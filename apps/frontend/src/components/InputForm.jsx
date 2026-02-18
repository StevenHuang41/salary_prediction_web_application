import { useEffect, useRef, useState } from "react";
import { getUniqJobTitle, retrainModel } from "../api/dataService";
import SelectInput from "./SelectInput";
import TermsCheckbox from "./TermsCheckbox";
import AgeYearsModal from "./AgeYearsModal";

const InputForm = ({
  getSubmitData,
  handleInputFormSubmit,
  setPredictResult,
  addToast,
  loadingFunc,
  setLoadingFunc,
  setErrFunc,
  dataAdded,
  setDataAdded,
}) => {
  const formRef = useRef(null);

  const [yearValid, setYearValid] = useState(true);
  const [age, setAge] = useState('');
  const [gender, setGender] = useState('');
  const [educationLevel, setEducationLevel] = useState('');
  const [jobTitle, setJobTitle] = useState('');
  const [yearE, setYearE] = useState('');
  // const [age, setAge] = useState('26');
  // const [gender, setGender] = useState('female');
  // const [educationLevel, setEducationLevel] = useState('Master');
  // const [jobTitle, setJobTitle] = useState('Data Scientist');
  // const [yearE, setYearE] = useState('8');
  const [jobOptionsLoading, setJobOptionsLoading] = useState(true);

  const ageYearModalTrigger = 
    document.getElementById('ageYearModalTrigger');

  const handleSubmit = (e) => {
    e.preventDefault();
    e.stopPropagation();

    // const forms = formRef.current;
    formRef.current.classList.add('was-validated');
    
    // check form has select value
    if (!formRef.current.checkValidity()) {
      setPredictResult(false);
      return;
    }
    
    // check age - year is not lower than 18
    // if ((age - yearE) < 18) {
    //   console.log('18!!!');
      
    //   setYearE('');
    //   ageYearModalTrigger.click();
    //   setYearValid(false);
    //   setPredictResult(false);
    //   return 
    // }

    // set data
    const data = {
      age: age,
      gender: gender,
      education_level: educationLevel,
      job_title: jobTitle,
      years_of_experience: yearE,
    } 
    getSubmitData(data);
    handleInputFormSubmit();
  };

  const handleChange = (e) => {
    const name = e.target.id;
    
    if (name === 'yearESelectInput') {
      if ((age - e.target.value) < 18) {
        formRef.current.classList.add('was-validated');
        setYearE('');
        ageYearModalTrigger.click();
        setYearValid(false);
        // setPredictResult(false);
        return;
      }
    }
    setPredictResult(false);
  };

  const ageOptions = Array.from({ length: 71 }, (_, i) => (
    {value: i + 18, text: i + 18}
  ));

  const yearEOptions = Array.from({ length: 71 }, (_, i) => (
    {value: i, text: i}
  ));

  // update data when form changes
  useEffect(() => {
    getSubmitData({
      age: age,
      gender: gender,
      education_level: educationLevel,
      job_title: jobTitle,
      years_of_experience: yearE,
    });
  }, [age, gender, educationLevel, jobTitle, yearE, getSubmitData]);

  // get job title options
  const [jobOptions, setJobOptions] = useState([]);
  useEffect(() => {
    const abortController = new AbortController();
    const getData = async () => {
      try {
        const data = await getUniqJobTitle();
        const options = data.value.map((val) => (
          {value: val, text: val}
        ));
        setJobOptions(options);
        setJobOptionsLoading(false);
      } catch (err) {console.log(err);}
    };
    getData();
    return () => abortController.abort()
  }, []);


  // handle retrain btn click
  const handleRetrain = async () => {
    setErrFunc(null);
    addToast("Retrain Model ...", "warning")
    
    try {
      setLoadingFunc(true);
      const res = await retrainModel({
        age: age,
        gender: gender,
        education_level: educationLevel,
        job_title: jobTitle,
        years_of_experience: yearE,
      });
      setPredictResult(res.result)
      // console.log(res.message);
      addToast("Retrain model successfully!", "success")
      setDataAdded(false);
    } catch (err) {
      setErrFunc(err.message);
      addToast("Retrain model failed!", "danger")
    } finally {
      setLoadingFunc(false);
    }
  };

  return (<>
    {/* headline */}
    <div className="row">
      <div className="col">
        <div className="text-primary fs-1">
          Salary Prediction
        </div>
      </div>
    </div>

    {/* form */}
    <form
      id="InputForm"
      className={`needs-validation`}
      noValidate
      ref={formRef}
      onSubmit={handleSubmit}
      onChange={handleChange}
    >
      <div className={`row row-cols-1 row-cols-md-2 g-2`} >

        <SelectInput
          className="col col-xl-2"
          selectId='ageSelectInput'
          options={ageOptions}
          invalidFeedbackText='Please select a valid age.'
          value={age}
          onChange={e => setAge(e.target.value)}
          isLoadingOptions={false}
        >
          Age
        </SelectInput>


        <SelectInput
          className="col col-xl-2"
          selectId='genderSelectInput'
          options={[
            {value: 'male', text: 'Male'},
            {value: 'female', text: 'Female'},
            {value: 'other', text: 'Other'},
          ]}
          invalidFeedbackText='Please select a gender.'
          value={gender}
          onChange={e => setGender(e.target.value)}
          isLoadingOptions={false}
        >
          Gender
        </SelectInput>


        <SelectInput
          selectId='eduLevSelectInput'
          options={[
            {value: 'No specified', text: 'No specified'},
            {value: 'High School', text: 'High School'},
            {value: 'Bachelor', text: 'Bachelor'},
            {value: 'Master', text: 'Master'},
            {value: 'PhD', text: 'PhD'},
          ]}
          invalidFeedbackText='Please select an education level.'
          className="col col-xl-3"
          value={educationLevel}
          onChange={e => setEducationLevel(e.target.value)}
          isLoadingOptions={false}
        >
          Education level
        </SelectInput>

        <SelectInput
          selectId='jobTitleSelectInput'
          options={jobOptions}
          invalidFeedbackText='Please select a job title.'
          className="col col-xl-3"
          value={jobTitle}
          onChange={e => setJobTitle(e.target.value)}
          isLoadingOptions={jobOptionsLoading}
        >
          Job title
        </SelectInput>

        <SelectInput
          selectId='yearESelectInput'
          options={yearEOptions}
          invalidFeedbackText={
            yearValid ? 'Please select a valid number.' :
            `The years of experience should not exceed ${age - 18}.`
          }
          className="col col-xl-2"
          value={yearE}
          onChange={e => setYearE(e.target.value)}
          isLoadingOptions={false}
        >
          Years of experience
        </SelectInput>
      </div>

      <div
        className={`
          row row-cols-1 row-cols-md-2
          mx-0 mt-1
          d-flex
          align-items-center
        `}
      >

        <TermsCheckbox 
          className={`
            col
            p-0 my-2 
          `}
          modalId='termsModal'
          labelText='Agree to'
          btnText='terms and conditions'
          invalidFeedbackText='You must agree before submitting.'
          setPredictResult={setPredictResult}
        />

        <div
          className={`
            col
            m-0 p-0 
            d-flex
            align-items-center
            justify-content-md-end
          `}
        >
          {dataAdded &&
          <div
            className={`
              btn btn-outline-warning
              /p-2 /py-1 me-1
              text-nowrap
              ${loadingFunc && `disabled`}
            `}
            onClick={handleRetrain}
          >
            Retrain Model
          </div>
          }
          <button
            className={`
              btn btn-primary
              ${loadingFunc && `disabled`}
            `}
            type="submit"
            id="predictSalaryBtn"
          >
            Predict Salary
          </button>
        </div>

      </div>
    </form>


    <button
      id='ageYearModalTrigger'
      data-bs-toggle="modal"
      data-bs-target={'#ageYearModal'}
      style={{display: 'none'}}
    />

    <AgeYearsModal
      id="ageYearModal"
    />



  </>)
};

export default InputForm;