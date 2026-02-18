import ModalTemplate from "./ModalTemplate";

const AgeYearsModal = ({
  id,
}) => {
  return (
    <ModalTemplate
      id={id}
      modalHeaderText={`
        Invalid Age and Years of Experience
      `}
      modalBodyText={`
        The result of Age - Years of
        experience must not less than 18.
      `}
      hasSecondaryBtn={false}
      primaryText={'Ok'}
      cName={'modal-dialog-centered'}

    />
  )
}

export default AgeYearsModal;