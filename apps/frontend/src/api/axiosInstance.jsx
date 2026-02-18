import axios from 'axios';

const api0 = axios.create({
  // depends on frontend address
  baseURL: `${import.meta.env.VITE_IP_ADDRESS}:8000/api`,
  headers: {
    'Content-Type': 'application/json'
  }
});

const api1 = axios.create({
  baseURL: `${import.meta.env.VITE_IP_ADDRESS}:8001/api`,
  headers: {
    'Content-Type': 'application/json'
  }
});

export { api0, api1 };