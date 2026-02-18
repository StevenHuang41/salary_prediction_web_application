import { render, screen, fireEvent, } from '@testing-library/react';
import SelectInput from '../SelectInput'
import { vi, describe, expect, it } from 'vitest';

vi.mock('../LoadingResult', () => ({
  default: () => (
    <div data-testid="loading-container">loading</div>
  ) }));

describe('SelectInput', () => {
  const baseProps = {
    selectId: 'test-select',
    options: [
      { value: '1', text: 'Option 1'},
      { value: '2', text: 'Option 2'},
    ],
    invalidFeedbackText: "Invalid field",
    className: 'custom-class',
    defaultValue: '',
    value: '',
    onChange: vi.fn(),
    children: 'col',
    isLoadingOptions: false,
  };

  it('renders label and options', () => {
    render(<SelectInput {...baseProps} />);
    expect(screen.queryByText(baseProps.children)).toBeInTheDocument();
    expect(screen.getByText('Option 1')).toBeInTheDocument();
    expect(screen.getByText('Option 2')).toBeInTheDocument();
  });

  it('show correct placeholder', () => {
    render(<SelectInput {...baseProps} />);
    expect(screen.getByText(`Choose ${(baseProps.children).toLowerCase()}`))
    .toBeInTheDocument();
  });

  it('apply custom className', () => {
    render(<SelectInput {...baseProps} />);
    const container = document.querySelector('.form-floating');
    expect(container).toHaveClass('custom-class');
  });

  it('apply null className', () => {
    render(<SelectInput {...baseProps} className="" />);
    const container = document.querySelector('.form-floating');
    expect(container).not.toHaveClass('custom-class');
  });

  it('call onChange when selecting option', () => {
    render(<SelectInput {...baseProps} />);
    const select = screen.getByLabelText(baseProps.children);
    fireEvent.change(select, { target: { value: '2' } });
    expect(baseProps.onChange).toHaveBeenCalled();
  });

  it('show invalid feedback text', () => {
    render(<SelectInput {...baseProps} />);
    expect(screen.getByText(baseProps.invalidFeedbackText))
    .toBeInTheDocument();
  });

  it('show loading indicator when loading', () => {
    render(<SelectInput {...baseProps} isLoadingOptions={true} />);
    expect(screen.getByTestId('loading-container')).toBeInTheDocument();
    expect(screen.queryByText(baseProps.children)).not.toBeInTheDocument();
  });
  
  it('apply the right className when value is empty', () => {
    render(<SelectInput {...baseProps} />);
    const select = screen.getByLabelText('col');
    expect(select).toHaveClass('text-secondary');
    expect(select).not.toHaveClass('fw-bold');
  });

  it('apply the right className when value is not empty', () => {
    render(<SelectInput {...baseProps} value={1} />);
    const select = screen.getByLabelText('col')
    expect(select).toHaveClass('fw-bold');
    expect(select).not.toHaveClass('text-secondary');
  });

  it('correct selectId', () => {
    render(<SelectInput {...baseProps} />);
    const select = document.querySelector('.form-select')
    expect(select).toHaveAttribute('id', baseProps.selectId);
  });
});