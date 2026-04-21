import { ChangeDetectionStrategy, Component, Input, computed, signal } from '@angular/core';
import { CommonModule } from '@angular/common';

import { ShapContribution } from '../../core/models/prediction.model';

@Component({
  selector: 'app-shap-bar',
  standalone: true,
  imports: [CommonModule],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="space-y-2">
      @for (row of rows(); track row.feature) {
        <div class="grid grid-cols-12 items-center gap-3 text-xs">
          <div class="col-span-4 truncate text-surface-200" [title]="row.feature">
            {{ row.feature }}
          </div>
          <div class="col-span-7 relative h-3 rounded-full bg-surface-800 overflow-hidden">
            <div class="absolute inset-y-0 left-1/2 w-px bg-surface-600/60"></div>
            <div
              class="absolute inset-y-0 transition-all"
              [class.bg-status-dropout]="row.contribution > 0"
              [class.bg-status-graduate]="row.contribution <= 0"
              [style.left.%]="row.contribution > 0 ? 50 : 50 - row.widthPct"
              [style.width.%]="row.widthPct"
            ></div>
          </div>
          <div
            class="col-span-1 text-right tabular-nums font-medium"
            [class.text-status-dropout]="row.contribution > 0"
            [class.text-status-graduate]="row.contribution <= 0"
          >
            {{ row.contribution > 0 ? '+' : '' }}{{ row.contribution.toFixed(2) }}
          </div>
        </div>
      }
      @if (rows().length === 0) {
        <p class="text-xs text-surface-400 italic">No contributions to display.</p>
      }
    </div>
  `,
})
export class ShapBarComponent {
  private readonly _features = signal<ShapContribution[]>([]);

  @Input({ required: true }) set features(values: ShapContribution[]) {
    this._features.set(values ?? []);
  }
  @Input() limit = 8;

  readonly rows = computed(() => {
    const features = this._features().slice(0, this.limit);
    if (!features.length) {
      return [];
    }
    const max = Math.max(...features.map((f) => Math.abs(f.contribution)), 0.0001);
    return features.map((f) => ({
      ...f,
      widthPct: Math.min(50, (Math.abs(f.contribution) / max) * 50),
    }));
  });
}
