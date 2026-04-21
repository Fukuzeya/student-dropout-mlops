import { ChangeDetectionStrategy, Component, Input } from '@angular/core';
import { NgClass } from '@angular/common';

export type KpiTrend = 'up' | 'down' | 'flat';
export type KpiTone = 'neutral' | 'positive' | 'warning' | 'negative' | 'brand';

@Component({
  selector: 'app-kpi-card',
  standalone: true,
  imports: [NgClass],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <article class="surface-card p-5 flex flex-col gap-3" [attr.aria-label]="label">
      <header class="flex items-start justify-between gap-3">
        <div>
          <p class="text-2xs uppercase tracking-wider text-surface-400">{{ label }}</p>
          <p class="mt-1 text-3xl font-semibold text-white tabular-nums">{{ value }}</p>
        </div>
        @if (icon) {
          <div
            class="h-9 w-9 rounded-lg flex items-center justify-center text-sm font-semibold"
            [ngClass]="iconBgClass()"
          >
            {{ icon }}
          </div>
        }
      </header>
      @if (subtitle || delta) {
        <footer class="flex items-center justify-between text-xs">
          <span class="text-surface-400">{{ subtitle }}</span>
          @if (delta) {
            <span class="inline-flex items-center gap-1 font-medium" [ngClass]="trendClass()">
              <span>{{ trendArrow() }}</span>
              <span class="tabular-nums">{{ delta }}</span>
            </span>
          }
        </footer>
      }
    </article>
  `,
})
export class KpiCardComponent {
  @Input({ required: true }) label!: string;
  @Input({ required: true }) value: string | number = '—';
  @Input() subtitle?: string;
  @Input() icon?: string;
  @Input() tone: KpiTone = 'neutral';
  @Input() trend: KpiTrend = 'flat';
  @Input() delta?: string;

  iconBgClass(): string {
    switch (this.tone) {
      case 'positive':
        return 'bg-status-graduate-bg text-status-graduate ring-1 ring-status-graduate-ring';
      case 'warning':
        return 'bg-status-enrolled-bg text-status-enrolled ring-1 ring-status-enrolled-ring';
      case 'negative':
        return 'bg-status-dropout-bg text-status-dropout ring-1 ring-status-dropout-ring';
      case 'brand':
        return 'bg-brand-600/15 text-brand-300 ring-1 ring-brand-500/40';
      default:
        return 'bg-surface-800 text-surface-200 ring-1 ring-surface-700';
    }
  }

  trendClass(): string {
    if (this.trend === 'up') {
      return this.tone === 'negative' ? 'text-status-dropout' : 'text-status-graduate';
    }
    if (this.trend === 'down') {
      return this.tone === 'negative' ? 'text-status-graduate' : 'text-status-dropout';
    }
    return 'text-surface-300';
  }

  trendArrow(): string {
    if (this.trend === 'up') return '▲';
    if (this.trend === 'down') return '▼';
    return '—';
  }
}
