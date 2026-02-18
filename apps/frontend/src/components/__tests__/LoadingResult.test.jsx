import {
  render,
  screen,
} from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import LoadingResult from '../LoadingResult';

describe('LoadingResult', () => {
  const baseProps = {
    setClass: 'container-class',
    setStyle: 'container-style',
    setTextClass: 'text-class',
    loadingText: 'loading text'
  };

  it('apply className on container', () => {
    render(<LoadingResult {...baseProps}/>);
    const container = document.querySelector('div[style]');
    expect(container).toHaveClass(baseProps.setClass);
  });

  it('apply style on container', () => {
    render(<LoadingResult {...baseProps}/>);
    const container = document.querySelector('div[style');
    expect(container).toHaveClass(baseProps.setClass);
  });

  it('apply className on text div', () => {
    render(<LoadingResult {...baseProps}/>);
    const textDiv = document.querySelector('div#loading-text');
    expect(textDiv).toHaveClass(baseProps.setTextClass)
  });

  it('correct loading text', () => {
    render(<LoadingResult {...baseProps}/>);
    expect(screen.getByText(baseProps.loadingText)).toBeInTheDocument();
  });
});