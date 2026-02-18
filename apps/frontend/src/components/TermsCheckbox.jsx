import { useState } from 'react'
import TermsModal from './TermsModal'

const TermsCheckbox = ({
  className,
  modalId,
  labelText,
  btnText,
  invalidFeedbackText,
  setPredictResult
}) => {
  const [agree, setAgree] = useState(false);

  return (<>
    <div className={`${className || ''}`}>
      <div className={`form-check`}>
        <input
          className="form-check-input"
          type="checkbox"
          checked={agree}
          id="invalidCheck"
          onChange={(e) => setAgree(e.target.checked)}
          required
        />
        <label
          className={`
            form-check-label
            col-auto ms-1
            d-flex
            align-items-center
          `}
          htmlFor="invalidCheck"
        >
          {labelText}
          <button
            className={`
              btn btn-link
              col-auto
              p-0 ps-1
            `}
            data-bs-toggle="modal"
            data-bs-target={`#${modalId}`}
            onClick={(e) => {e.preventDefault()}}
          >
            {btnText}
          </button>
        </label>
        <div className="invalid-feedback">
          {invalidFeedbackText}
        </div>

      </div>
    </div>

    <TermsModal
      id={modalId}
      handleModalSecondaryClick={() => {
        setAgree(false);
        setPredictResult(false);
      }}
      handleModalPrimaryClick={() => (
        setAgree(true)
      )}
    />
  </>)
}

export default TermsCheckbox;