
const ModalTemplate = ({
  id,
  cName,
  modalHeaderText,
  modalBodyText,

  handleModalPrimaryClick,
  primaryText,

  hasSecondaryBtn,
  handleModalSecondaryClick,
  secondaryText,
}) => {
  return (
    <div
      className="modal fade"
      id={id}
      tabIndex="-1"
      aria-hidden="true"
    >
      <div className={`modal-dialog ${cName}`}>
        <div className="modal-content">
          <div className="modal-header">
            <h1 className="modal-title fs-5">
              {modalHeaderText}
            </h1>
            <button
              type="button"
              className="btn-close"
              data-bs-dismiss="modal"
              aria-label="Close"
            />
          </div>

          <div className="modal-body">
            {modalBodyText}
          </div>

          <div className="modal-footer">
            {hasSecondaryBtn && <button
              type="button"
              className="btn btn-secondary"
              data-bs-dismiss="modal"
              onClick={handleModalSecondaryClick}
            >
              {secondaryText}
            </button>}
            <button
              type="button"
              className="btn btn-primary"
              data-bs-dismiss="modal"
              onClick={handleModalPrimaryClick}
            >
              {primaryText}
            </button>
          </div>

        </div>
      </div>
    </div>
  )
}

export default ModalTemplate;