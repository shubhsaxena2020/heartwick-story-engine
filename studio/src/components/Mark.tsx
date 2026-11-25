/** Brand mark: a candle flame (the "wick") over a hearth line. Original artwork. */
export function Mark({ size = 30 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true" className="shrink-0">
      <rect x="0.5" y="0.5" width="31" height="31" rx="4" fill="#15120f" stroke="#3d342c" />
      <path d="M16 6c3 4 6 6.5 6 11a6 6 0 0 1-12 0c0-3 2-4.5 3-7 1 2 2 3 3 3-1-3-1-5 0-7z" fill="var(--color-sodium)" />
      <path d="M16 15c1.4 1.8 2.4 3 2.4 4.6a2.4 2.4 0 0 1-4.8 0c0-1.6 1-2.8 2.4-4.6z" fill="var(--color-ink)" opacity=".92" />
      <path d="M9 27h14" stroke="var(--color-sodium-deep)" strokeWidth="2" />
    </svg>
  );
}
