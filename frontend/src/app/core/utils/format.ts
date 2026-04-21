import { RiskLevel } from '../models/prediction.model';

export const RISK_ORDER: Record<RiskLevel, number> = { High: 0, Medium: 1, Low: 2 };

export function pct(value: number, digits = 1): string {
  if (!Number.isFinite(value)) {
    return '—';
  }
  return `${(value * 100).toFixed(digits)}%`;
}

export function num(value: number, digits = 0): string {
  if (!Number.isFinite(value)) {
    return '—';
  }
  return value.toLocaleString(undefined, { maximumFractionDigits: digits });
}

export function timeAgo(iso: string | null | undefined): string {
  if (!iso) {
    return '—';
  }
  const ts = new Date(iso).getTime();
  if (Number.isNaN(ts)) {
    return '—';
  }
  const seconds = Math.max(0, Math.floor((Date.now() - ts) / 1000));
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

export function riskLabel(level: RiskLevel): string {
  return level;
}
