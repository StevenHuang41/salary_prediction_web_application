import { describe, it, expect, vi, beforeEach } from "vitest";

beforeEach(() => {
  vi.resetModules();
});

describe('axiosInstance', () => {
  it('create api0 with correct baseURL', async () => {
    import.meta.env.VITE_IP_ADDRESS = 'http://localhost0';

    const { api0 } = await import("../axiosInstance");
    expect(api0.defaults.baseURL).toBe('http://localhost0:8000/api');
  });

  it('create api1 with correct baseURL', async () => {
    import.meta.env.VITE_IP_ADDRESS = 'http://localhost1';

    const { api1 } = await import("../axiosInstance");
    expect(api1.defaults.baseURL).toBe('http://localhost1:8001/api');
  });
});