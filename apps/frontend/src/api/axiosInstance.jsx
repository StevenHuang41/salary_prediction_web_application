import axios from 'axios';

const api0 = axios.create({
  // depends on frontend address
  baseURL: `${import.meta.env.VITE_API_BASE_URL}/api/v1`,
  headers: {
    'Content-Type': 'application/json'
  }
});

export { api0 };
