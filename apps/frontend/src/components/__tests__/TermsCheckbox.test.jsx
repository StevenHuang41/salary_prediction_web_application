import {
  render,
  screen,
  fireEvent,
} from '@testing-library/react';
import { vi, describe, expect, it, beforeEach } from 'vitest';
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
    setPredictResult: vi.fn(),
  };

  beforeEach(() => {
    baseProps.setPredictResult.mockClear();
  });

  it('render checkbox, text, modal button and feedback', () => {
    render(<TermsCheckbox {...baseProps} />);
    expect(document.querySelector('input#invalidCheck')).toBeInTheDocument();
    expect(screen.getByText(baseProps.labelText)).toBeInTheDocument();
    expect(screen.getByText(baseProps.btnText)).toBeInTheDocument();
    expect(screen.getByText(baseProps.invalidFeedbackText)).toBeInTheDocument();
  });

  it('checkbox clickable', () => {
    render(<TermsCheckbox {...baseProps} />);
    const checkbox = document.querySelector('input#invalidCheck');
    expect(checkbox).not.toBeChecked();
    fireEvent.click(checkbox);
    expect(checkbox).toBeChecked();
    fireEvent.click(checkbox);
    expect(checkbox).not.toBeChecked();
  });

  it('modal button has correct id target that matches to modal', () => {
    render(<TermsCheckbox {...baseProps} />);
    const modalBtn = document.querySelector('button.btn-link');
    const modal = screen.getByTestId('TermsModal');
    expect(modalBtn).toHaveAttribute('data-bs-toggle', 'modal');
    expect(modalBtn).toHaveAttribute('data-bs-target', `#${baseProps.modalId}`);
    expect(modal).toHaveAttribute('id', baseProps.modalId);
  });

  it('clicking agree in modal will check the checkbox, and vice versa', () => {
    render(<TermsCheckbox {...baseProps} />);
    const agreeBtn = screen.getByText('Agree');
    const disagreeBtn = screen.getByText('Disagree');
    const checkbox = document.querySelector('input.form-check-input');
    expect(checkbox).not.toBeChecked();

    fireEvent.click(agreeBtn);
    expect(checkbox).toBeChecked();

    fireEvent.click(disagreeBtn);
    expect(checkbox).not.toBeChecked();
    expect(baseProps.setPredictResult).toHaveBeenCalledWith(false);
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