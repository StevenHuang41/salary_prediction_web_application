import ModalTemplate from "./ModalTemplate";

const TermsModal = ({
  id,
  handleModalSecondaryClick,
  handleModalPrimaryClick,
}) => {

  const modalBodyText = ('If you agree with these terms and conditions,' +
    ' you authorize our team to collect your data for future improvement.' +
    ' Your data will be used to train backend machine learning models,' + 
    ' and only for training purpose.')

  return (<>
    <ModalTemplate
      id={id}
      modalHeaderText="Terms and Conditions"
      modalBodyText={modalBodyText}
      hasSecondaryBtn={true}
      handleModalSecondaryClick={handleModalSecondaryClick}
      secondaryText='Disagree'
      handleModalPrimaryClick={handleModalPrimaryClick}
      primaryText='Agree'
    />

  </>)
};

export default TermsModal;