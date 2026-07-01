import '@testing-library/jest-dom';

// Mock ResizeObserver which is used by Recharts
global.ResizeObserver = class ResizeObserver {
  constructor(cb) {
    this.cb = cb;
  }
  observe() {}
  unobserve() {}
  disconnect() {}
};
