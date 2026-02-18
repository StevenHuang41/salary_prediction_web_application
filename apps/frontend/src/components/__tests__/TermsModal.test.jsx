import {
  render,
  screen,
  fireEvent,
} from '@testing-library/react';
import { vi, describe, expect, it } from 'vitest';
import TermsModal from '../TermsModal';

vi.mock('../ModalTemplate', () => ({
  default: ({
    id,
    modalHeaderText,
    modalBodyText,
    hasSecondaryBtn,
    handleModalSecondaryClick,
    secondaryText,
    handleModalPrimaryClick,
    primaryText,
  }) => (
    <div data-testid="modal-container">
      <div>{modalHeaderText}</div>
      <div>{modalBodyText}</div>
      <button onClick={handleModalSecondaryClick}>{secondaryText}</button>
      <button onClick={handleModalPrimaryClick}>{primaryText}</button>
      <div data-testid="modal-id">{id}</div>
      <div data-testid="modal-hasSecondaryBtn">{hasSecondaryBtn.toString()}</div>
    </div>
  ),
}));

describe('TermsModal', () => {
  const baseProps = {
    id: 'test-id',
    handleModalSecondaryClick: vi.fn(),
    handleModalPrimaryClick: vi.fn(),
  };

  it('correct id', () => {
    render(<TermsModal {...baseProps} />);
    expect(screen.getByText(baseProps.id)).toBeInTheDocument();
  });

  it('correct header and body text', () => {
    render(<TermsModal {...baseProps} />);
    expect(screen.getByText("Terms and Conditions")).toBeInTheDocument();
    expect(screen.getByText(/If you agree with these terms/i))
    .toBeInTheDocument();
  });

  it('has second button', () => {
    render(<TermsModal {...baseProps} />);
    expect(screen.getByText("true")).toBeInTheDocument();
  });

  it('has two correct text button', () => {
    render(<TermsModal {...baseProps} />);
    const agreeBtn = screen.getByText('Agree');
    const disagreeBtn = screen.getByText('Disagree');
    expect(agreeBtn).toBeInTheDocument();
    expect(disagreeBtn).toBeInTheDocument();
  });

  it('primary button click function', () => {
    render(<TermsModal {...baseProps} />);
    const agreeBtn = screen.getByText('Agree');
    fireEvent.click(agreeBtn);
    expect(baseProps.handleModalPrimaryClick).toHaveBeenCalled(1);
  });

  it('secondary button click function', () => {
    render(<TermsModal {...baseProps} />);
    const disagreeBtn = screen.getByText('Disagree');
    fireEvent.click(disagreeBtn);
    expect(baseProps.handleModalSecondaryClick).toHaveBeenCalled(1);
  });
});