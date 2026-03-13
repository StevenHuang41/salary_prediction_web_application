import { useEffect, useRef, useState } from "react";
import { getUniqJobTitle } from "../api/dataService";
import SelectInput from "./SelectInput";
import TermsCheckbox from "./TermsCheckbox";
import AgeYearsModal from "./AgeYearsModal";

const InputForm = ({
  onSubmit,
  setPredictState,
  setFormData,
  isPredicting,
}) => {
  const formRef = useRef(null);

  const [jobOptions, setJobOptions] = useState([]);

  const [loadJobState, setLoadJobState] = useState("loading");

  // production
  const [age, setAge] = useState('');
  const [gender, setGender] = useState('');
  const [educationLevel, setEducationLevel] = useState('');
  const [jobTitle, setJobTitle] = useState('');
  const [yearE, setYearE] = useState('');
  // // test
  // const [age, setAge] = useState('26');
  // const [gender, setGender] = useState('female');
  // const [educationLevel, setEducationLevel] = useState('Master');
  // const [jobTitle, setJobTitle] = useState('Data Scientist');
  // const [yearE, setYearE] = useState('0');
  //
  const yearValid =
    age === "" || yearE === "" || (Number(age) - Number(yearE)) >= 18;

  // get job title
  useEffect(() => {
    const getData = async () => {
      try {
        const data = await getUniqJobTitle();

        if (!data || data.length === 0) {
          return;
        }

        setJobOptions(data.map((val) => ({value: val, text: val})));
        setLoadJobState("success");
      } catch (err) {
        console.error(err);
        setLoadJobState("error");
      }
    };
    getData();
  }, []);

  // update formData when selections change
  useEffect(() => {
    setFormData({
      age: age,
      gender: gender,
      education_level: educationLevel,
      job_title: jobTitle,
      years_of_experience: yearE,
    });

    setPredictState({
      data: null,
      loading: null,
      error: null,
    });

    // TODO:
    console.log(
      `age ${age}`,
      `gender ${gender}`,
      `educationLevel ${educationLevel}`,
      `jobTitle ${jobTitle}`,
      `yearE ${yearE}`,
    );

  }, [age, gender, educationLevel, jobTitle, yearE]);

  // submit
  const handleSubmit = (e) => {
    e.preventDefault();

    if (!yearValid) {
      setYearE("");
      return;
    }

    formRef.current.classList.add('was-validated');

    if (!formRef.current.checkValidity()) {
      setPredictState({
        data: null,
        loading: null,
        error: null,
      });
      return;
    }

    onSubmit();
  };

  const ageOptions = Array.from({ length: 82 }, (_, i) => ({
      value: i + 18,
      text: i + 18
  }));

  const yearEOptions = Array.from({ length: 82 }, (_, i) => ({
      value: i,
      text: i
  }));

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
      className="needs-validation"
      noValidate
      ref={formRef}
      onSubmit={handleSubmit}
    >
      <div className="row row-cols-1 row-cols-md-2 g-2">

        <SelectInput
          className="col col-xl-2"
          selectId="ageSelectInput"
          options={ageOptions}
          value={age}
          onChange={(e) => setAge(e.target.value)}
          invalidFeedbackText="Please select a valid age."
        >
          Age
        </SelectInput>


        <SelectInput
          className="col col-xl-2"
          selectId="genderSelectInput"
          options={[
            {value: 'male', text: 'Male'},
            {value: 'female', text: 'Female'},
            {value: 'other', text: 'Other'},
          ]}
          value={gender}
          onChange={(e) => setGender(e.target.value)}
          invalidFeedbackText="Please select a gender."
        >
          Gender
        </SelectInput>


        <SelectInput
          className="col col-xl-3"
          selectId="eduLevSelectInput"
          options={[
            {value: 'No specified', text: 'No specified'},
            {value: 'High School', text: 'High School'},
            {value: 'Bachelor', text: 'Bachelor'},
            {value: 'Master', text: 'Master'},
            {value: 'PhD', text: 'PhD'},
          ]}
          value={educationLevel}
          onChange={(e) => setEducationLevel(e.target.value)}
          invalidFeedbackText="Please select an education level."
        >
          Education level
        </SelectInput>

        <SelectInput
          className="col col-xl-3"
          selectId="jobTitleSelectInput"
          options={jobOptions}
          value={jobTitle}
          onChange={(e) => setJobTitle(e.target.value)}
          invalidFeedbackText='Please select a job title.'
          isLoadingOptions={loadJobState === "loading"}
        >
          Job title
        </SelectInput>

        <SelectInput
          className="col col-xl-2"
          selectId="yearESelectInput"
          options={yearEOptions}
          value={yearE}
          onChange={(e) => setYearE(e.target.value)}
          isValid={yearValid}
          invalidFeedbackText={
            yearValid
              ? "Please select a valid number."
              : `The years of experience should not exceed ${age - 18}.`
          }
        >
          Years of experience
        </SelectInput>
      </div>

      <div
        className={`
          row row-cols-1 row-cols-md-2
          mx-0 mt-1
          d-flex align-items-center
        `}
      >

        <TermsCheckbox
          className="col p-0 my-2"
          modalId="termsModal"
          labelText="Agree to"
          btnText="terms and conditions"
          invalidFeedbackText="You must agree before submitting."
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
          <button
            className="btn btn-primary"
            type="submit"
            id="predictSalaryBtn"
            data-testid="predictSalaryBtn"
            disabled={isPredicting}
          >
            {isPredicting ? "Predicting ..." : "Predict Salary"}
          </button>
        </div>

      </div>
    </form>

    <button
      id="ageYearModalTrigger"
      data-bs-toggle="modal"
      data-bs-target={"#ageYearModal"}
      style={{display: "none"}}
    />

    <AgeYearsModal id="ageYearModal" />
  </>)
};

export default InputForm;
