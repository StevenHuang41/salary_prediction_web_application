import LoadingResult from "./LoadingResult";

const SelectInput = ({
  selectId,
  options,
  invalidFeedbackText,
  className,
  value,
  onChange,
  children,
  isLoadingOptions,
}) => {

  return (<>
    <div className={`form-floating ${className || ''}`}>
      {isLoadingOptions ? 
      <LoadingResult
        loadingText="Loading options"
        setStyle={{fontSize: "1.2em"}}
        setClass="m-1 m-md-3"
        setTextClass={`d-flex`}
      /> : <>
      <select
        className={`
          form-select fs-6
          ${value === '' ? 'text-secondary' : 'fw-bold'}
        `}
        id={selectId}
        value={value}
        onChange={onChange}
        required
      >
        <option value='' disabled>
          {!isLoadingOptions && `Choose ${children.toLowerCase()}`}
        </option>
        {options.map(({ value, text }) => (
          <option key={value} value={value}>{text}</option>
        ))}
      </select>
      <label htmlFor={selectId}>{children}</label>
      <div className='invalid-feedback'>
        {invalidFeedbackText}
      </div>
      </>}

    </div>
  </>)
};

export default SelectInput;