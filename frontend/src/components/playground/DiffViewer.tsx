'use client';

import React from 'react';

interface DiffViewerProps {
  answerA: string;
  answerB: string;
}

export default function DiffViewer({ answerA, answerB }: DiffViewerProps) {
  // Dynamic Programming LCS word-diff algorithm
  const diffWords = (oldStr: string, newStr: string) => {
    const oldWords = oldStr.split(/(\s+)/).filter(Boolean);
    const newWords = newStr.split(/(\s+)/).filter(Boolean);

    const dp: number[][] = Array(oldWords.length + 1)
      .fill(0)
      .map(() => Array(newWords.length + 1).fill(0));

    for (let i = 1; i <= oldWords.length; i++) {
      for (let j = 1; j <= newWords.length; j++) {
        if (oldWords[i - 1] === newWords[j - 1]) {
          dp[i][j] = dp[i - 1][j - 1] + 1;
        } else {
          dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
        }
      }
    }

    let i = oldWords.length;
    let j = newWords.length;
    const diffs: { type: 'added' | 'removed' | 'unchanged'; value: string }[] = [];

    while (i > 0 || j > 0) {
      if (i > 0 && j > 0 && oldWords[i - 1] === newWords[j - 1]) {
        diffs.push({ type: 'unchanged', value: oldWords[i - 1] });
        i--;
        j--;
      } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
        diffs.push({ type: 'added', value: newWords[j - 1] });
        j--;
      } else {
        diffs.push({ type: 'removed', value: oldWords[i - 1] });
        i--;
      }
    }

    return diffs.reverse();
  };

  const wordDiffs = diffWords(answerA, answerB);

  return (
    <div className="p-5 bg-white dark:bg-neutral-900 border border-neutral-100 dark:border-neutral-850 rounded-2xl shadow-sm space-y-4">
      <h3 className="text-sm font-bold text-neutral-400 uppercase tracking-wider">
        Comparative Diff Inspector (A → B)
      </h3>
      <div className="text-sm leading-relaxed whitespace-pre-wrap font-medium font-sans">
        {wordDiffs.map((part, index) => {
          if (part.type === 'added') {
            return (
              <span
                key={index}
                className="bg-emerald-100 dark:bg-emerald-950/40 text-emerald-800 dark:text-emerald-300 px-1 py-0.5 rounded transition-all"
                title="Added in Pipeline B"
              >
                {part.value}
              </span>
            );
          }
          if (part.type === 'removed') {
            return (
              <span
                key={index}
                className="bg-rose-105 dark:bg-rose-950/40 text-rose-800 dark:text-rose-300 px-1 py-0.5 rounded line-through decoration-rose-500 transition-all"
                title="Removed in Pipeline B"
              >
                {part.value}
              </span>
            );
          }
          return <span key={index}>{part.value}</span>;
        })}
      </div>
    </div>
  );
}
