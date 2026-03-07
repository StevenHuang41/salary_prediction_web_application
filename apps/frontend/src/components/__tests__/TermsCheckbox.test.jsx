import { render, screen, fireEvent, } from '@testing-library/react';
import { vi, describe, expect, it } from 'vitest';
import TermsCheckbox from '../TermsCheckbox';

vi.mock('../TermsModal', () => ({
  default: ({
    id,
    handleModalPrimaryClick,
    handleModalSecondaryClick,
  }) => (
    <div data-testid="TermsModal" id={id}>
      <button onClick={handleModalPrimaryClick}>Agree</button>
      <button onClick={handleModalSecondaryClick}>Disagree</button>
      <div>{id}</div>
    </div>
  )
}));

describe('TermsCheckbox', () => {
  const baseProps = {
    className: 'container-class',
    modalId: 'modal-id',
    labelText: 'label-text',
    btnText: 'button-text',
    invalidFeedbackText: 'Invalid feedback',
  };

  it('renders checkbox, text, modal button and feedback', () => {
    render(<TermsCheckbox {...baseProps} />);

    const checkbox = document.querySelector('input#invalidCheck')
    const checkboxText = screen.getByText(baseProps.labelText)
    const checkboxTBtn = screen.getByText(baseProps.btnText)
    const checkboxInvalidText = screen.getByText(baseProps.invalidFeedbackText)

    const modalBtn = document.querySelector('button.btn-link');
    const modal = screen.getByTestId('TermsModal');

    expect(checkbox).toBeInTheDocument();
    expect(checkboxText).toBeInTheDocument();
    expect(checkboxTBtn).toBeInTheDocument();
    expect(checkboxInvalidText).toBeInTheDocument();

    expect(modalBtn).toBeInTheDocument();
    expect(modal).toBeInTheDocument();

    expect(modalBtn).toHaveAttribute("data-bs-toggle", "modal");
    expect(modalBtn).toHaveAttribute("data-bs-target", "#modal-id");
    expect(modal).toHaveAttribute("id", baseProps.modalId)
  });

  it('toggles checkbox correctly', () => {
    render(<TermsCheckbox {...baseProps} />);

    const checkbox = document.querySelector('input#invalidCheck')
    const checkboxText = screen.getByText(baseProps.labelText)
    const checkboxTBtn = screen.getByText(baseProps.btnText)

    expect(checkbox).not.toBeChecked();

    fireEvent.click(checkbox);
    expect(checkbox).toBeChecked();

    fireEvent.click(checkbox);
    expect(checkbox).not.toBeChecked();

    fireEvent.click(checkboxText);
    expect(checkbox).toBeChecked();

    fireEvent.click(checkboxText);
    expect(checkbox).not.toBeChecked();

    fireEvent.click(checkboxTBtn);
    expect(checkbox).not.toBeChecked();
  });

  it('sets checkbox to checkend when modal "Agree" is clicked', () => {
    render(<TermsCheckbox {...baseProps} />);

    const checkbox = document.querySelector('input#invalidCheck')

    fireEvent.click(screen.getByText("Agree"))
    expect(checkbox).toBeChecked()

    fireEvent.click(screen.getByText("Disagree"))
    expect(checkbox).not.toBeChecked()
  });

  it('label is associated with checkbox', () => {
    render(<TermsCheckbox {...baseProps} />);

    const checkbox = document.querySelector('input#invalidCheck');
    const label = document.querySelector('label.form-check-label');

    expect(checkbox).toHaveAttribute('id', 'invalidCheck');
    expect(label).toHaveAttribute('for', 'invalidCheck');
  });
    
  it('correct className on container', () => {
    render(<TermsCheckbox {...baseProps} />);
    
    const wrapper = document.querySelector('div');
    const container = wrapper.firstChild;
    
    expect(container).toHaveClass(baseProps.className);
  });
  //
  it('correct null className on container', () => {
    render(<TermsCheckbox {...baseProps} className="" />);
    
    const wrapper1 = document.querySelector('div');
    const container1 = wrapper1.firstChild;
    
    expect(container1.className).toBe('');
  });
    
  it('click on modal button preventDefault', () => {
    render(<TermsCheckbox {...baseProps} />);
    const modalBtn = document.querySelector('button.btn-link');
    const mockEvent = vi.spyOn(window.Event.prototype, 'preventDefault');

    fireEvent.click(modalBtn);

    expect(mockEvent).toHaveBeenCalled();

    mockEvent.mockRestore();
  });
});
