'use client';

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useAppStore } from '../../store/appStore';
import { 
  Terminal, 
  GitBranch, 
  ShieldAlert, 
  FileText, 
  Sun, 
  Moon, 
  Cpu 
} from 'lucide-react';

export default function Sidebar() {
  const pathname = usePathname();
  const { theme, toggleTheme } = useAppStore();

  const navItems = [
    { name: 'Playground', path: '/playground', icon: Terminal },
    { name: 'Pipelines', path: '/pipelines', icon: GitBranch },
    { name: 'Evaluations', path: '/evaluations', icon: ShieldAlert },
    { name: 'Documents', path: '/documents', icon: FileText },
  ];

  return (
    <aside className="w-64 border-r border-neutral-200 dark:border-neutral-800 bg-white/70 dark:bg-neutral-900/70 backdrop-blur-md flex flex-col justify-between h-screen sticky top-0">
      <div className="flex flex-col">
        {/* Logo Section */}
        <div className="flex items-center gap-3 px-6 py-6 border-b border-neutral-100 dark:border-neutral-800/80">
          <div className="p-2 bg-gradient-to-tr from-cyan-500 to-blue-600 rounded-xl shadow-lg shadow-blue-500/20">
            <Cpu className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold bg-gradient-to-r from-blue-600 to-indigo-500 dark:from-cyan-400 dark:to-blue-500 bg-clip-text text-transparent leading-none">
              NeuroFlow
            </h1>
            <span className="text-[10px] text-neutral-400 font-semibold uppercase tracking-wider">
              Control Panel
            </span>
          </div>
        </div>

        {/* Navigation Items */}
        <nav className="mt-6 px-4 space-y-1.5">
          {navItems.map((item) => {
            const isActive = pathname === item.path;
            const Icon = item.icon;
            return (
              <Link
                key={item.path}
                href={item.path}
                className={`flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-300 font-medium ${
                  isActive
                    ? 'bg-blue-50 text-blue-600 dark:bg-blue-900/30 dark:text-cyan-400 border-l-4 border-blue-500 shadow-sm'
                    : 'text-neutral-500 hover:bg-neutral-50 dark:text-neutral-400 dark:hover:bg-neutral-800/50 hover:text-neutral-900 dark:hover:text-white'
                }`}
              >
                <Icon className={`w-5 h-5 ${isActive ? 'text-blue-500 dark:text-cyan-400' : 'text-neutral-400'}`} />
                <span>{item.name}</span>
              </Link>
            );
          })}
        </nav>
      </div>

      {/* Footer Section */}
      <div className="p-4 border-t border-neutral-100 dark:border-neutral-800/80 flex items-center justify-between">
        <span className="text-xs text-neutral-400 font-medium">v1.2.0</span>
        <button
          onClick={toggleTheme}
          className="p-2 rounded-xl border border-neutral-200 dark:border-neutral-800 hover:bg-neutral-50 dark:hover:bg-neutral-800/80 text-neutral-500 dark:text-neutral-400 transition-all duration-200"
          title="Toggle Theme"
        >
          {theme === 'dark' ? <Sun className="w-4 h-4 text-amber-500" /> : <Moon className="w-4 h-4 text-indigo-500" />}
        </button>
      </div>
    </aside>
  );
}
