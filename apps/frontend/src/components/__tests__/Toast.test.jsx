import { render, screen, waitFor } from '@testing-library/react';
import { vi, describe, it, expect } from 'vitest';
import Toast from '../Toast'
import userEvent from '@testing-library/user-event';

describe('Toast', () => {
  const baseProps = {
    toasts: [
      {id: 0, message: 'test0', showing: true, color: 'success'},
      {id: 1, message: 'test1', showing: false, color: 'warning'},
    ],
    removeToast: vi.fn(),
  };

  describe('Rendering', () => {
    it('render toast container when toasts = []', () => {
      render(<Toast {...baseProps} toasts={[]}/>);
      const toastContainer = document.querySelector('.toast-container');
      expect(toastContainer).toBeInTheDocument();
    });

    it('render correct number of toasts', () => {
      render(<Toast {...baseProps} />);
      const toastContainer = document.querySelector('.toast-container');
      expect((toastContainer.children).length).toEqual(baseProps.toasts.length);
    });
  });



  it('change className based on toast.showing', () => {
    render(<Toast {...baseProps} />);
    const firstToast = screen.getByText('test0');
    const secondToast = screen.getByText('test1');
    expect(firstToast).toBeInTheDocument();
    expect(secondToast).toBeInTheDocument();

    expect(firstToast.parentElement).toHaveClass('slide-in');
    expect(secondToast.parentElement).not.toHaveClass('slide-in');
    
    expect(firstToast.parentElement).not.toHaveClass('slide-out');
    expect(secondToast.parentElement).toHaveClass('slide-out');
  });

  it('call removeToast() when clicking the remove button', async () => {
    render(<Toast {...baseProps} />);
    const firstToast = screen.getByText('test0');
    const removeBtn = firstToast.nextSibling;
    expect(removeBtn).toBeInTheDocument();
    
    expect(baseProps.removeToast).not.toHaveBeenCalled();
    await userEvent.click(removeBtn);
    await waitFor(() => {
      expect(baseProps.removeToast).toHaveBeenCalled();
    });
  });
});