import { describe, it, expect, vi, beforeEach } from "vitest";

beforeEach(() => {
  vi.resetModules();
});

describe('axiosInstance', () => {
  it('create api0 with correct baseURL', async () => {
    import.meta.env.VITE_API_BASE_URL = 'http://localhost0:8080';

    const { api0 } = await import("../axiosInstance");
    expect(api0.defaults.baseURL).toBe('http://localhost0:8080/api/v1');
  });
});
