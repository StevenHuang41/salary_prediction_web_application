import { renderHook, act } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import useToast from '../useToast';

describe('useToast', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  })

  it('initial toast is an empty array', () => {
    const { result } = renderHook(() => useToast());
    expect(result.current.toasts).toHaveLength(0);
    // expect(result.current.toasts).toEqual([]);
  });

  it('add a toast', () => {
    const { result } = renderHook(() => useToast());
    
    act(() => {
      result.current.addToast('Test message', 'warning');
    });

    expect(result.current.toasts).toHaveLength(1);
    expect(result.current.toasts[0].showing).toBe(true);

    // after 3000ms
    act(() => vi.advanceTimersByTime(3000));

    // fade out animation
    expect(result.current.toasts).toHaveLength(1);
    expect(result.current.toasts[0].showing).toBe(false);

    // wait until fade out animation ends
    act(() => vi.advanceTimersByTime(500));

    expect(result.current.toasts).toHaveLength(0);
    expect(result.current.toasts).toEqual([]);
  });

  it('remove toast by clicking x button in toast', () => {
    const { result } = renderHook(() => useToast());

    act(() => {
      result.current.addToast('test message', 'success');
    });
    const toastId = result.current.toasts[0].id;

    // remove toast by clickign x button
    act(() => {
      result.current.removeToast(toastId);
    });

    // check if toast removed show fade out animation
    expect(result.current.toasts).toHaveLength(1);
    expect(result.current.toasts[0].showing).toBe(false);

    // wait until fade out animation ends
    act(() => vi.advanceTimersByTime(500));

    // check toast is removed from toasts
    expect(result.current.toasts).toHaveLength(0);
    expect(result.current.toasts).toEqual([]);
  });
  
  it('control multiple toasts independently', () => {
    const { result } = renderHook(() => useToast());
    act(() => {
      result.current.addToast('first toast', 'success');
    });
    const firstToast = result.current.toasts[0];

    act(() => {
      result.current.addToast('second toast', 'success');
    });

    // remove first toast
    act(() => {
      result.current.removeToast(firstToast.id);
    });

    // check exist two toasts
    expect(result.current.toasts).toHaveLength(2);
    expect(result.current.toasts[0].showing).toBe(false);
    expect(result.current.toasts[1].showing).toBe(true);

    // after clicking remove button
    act(() => vi.advanceTimersByTime(500));
    expect(result.current.toasts).toHaveLength(1);

    // after 500ms + 2500ms, secondToast
    act(() => vi.advanceTimersByTime(2500));
    expect(result.current.toasts[0].showing).toBe(false);
    act(() => vi.advanceTimersByTime(500));
    expect(result.current.toasts).toHaveLength(0);
    expect(result.current.toasts).toEqual([]);
  });









});
