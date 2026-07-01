'use client';

import React, { useEffect } from 'react';
import Sidebar from './Sidebar';
import { useAppStore } from '../../store/appStore';

export default function AppLayout({ children }: { children: React.ReactNode }) {
  const { theme } = useAppStore();

  useEffect(() => {
    const root = window.document.documentElement;
    root.classList.remove('light', 'dark');
    root.classList.add(theme);
  }, [theme]);

  return (
    <div className="flex bg-neutral-50 dark:bg-neutral-950 text-neutral-900 dark:text-neutral-50 min-h-screen transition-colors duration-200">
      <Sidebar />
      <main className="flex-1 overflow-y-auto max-h-screen p-8">
        <div className="max-w-7xl mx-auto w-full">
          {children}
        </div>
      </main>
    </div>
  );
}
