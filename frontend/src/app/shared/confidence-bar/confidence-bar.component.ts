import { ChangeDetectionStrategy, Component, Input } from '@angular/core';

import { ClassProbabilities } from '../../core/models/prediction.model';

@Component({
  selector: 'app-confidence-bar',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="space-y-1.5" [attr.aria-label]="'Class probability distribution'">
      <div class="flex h-2.5 rounded-full overflow-hidden bg-surface-800 ring-1 ring-surface-700">
        <div
          class="bg-status-dropout transition-all"
          [style.width.%]="pct(probabilities.Dropout)"
          [title]="'Dropout: ' + (probabilities.Dropout * 100).toFixed(1) + '%'"
        ></div>
        <div
          class="bg-status-enrolled transition-all"
          [style.width.%]="pct(probabilities.Enrolled)"
          [title]="'Enrolled: ' + (probabilities.Enrolled * 100).toFixed(1) + '%'"
        ></div>
        <div
          class="bg-status-graduate transition-all"
          [style.width.%]="pct(probabilities.Graduate)"
          [title]="'Graduate: ' + (probabilities.Graduate * 100).toFixed(1) + '%'"
        ></div>
      </div>
      @if (showLegend) {
        <div class="grid grid-cols-3 gap-2 text-2xs text-surface-300 tabular-nums">
          <span class="flex items-center gap-1">
            <span class="h-1.5 w-1.5 rounded-full bg-status-dropout"></span>
            Dropout {{ (probabilities.Dropout * 100).toFixed(1) }}%
          </span>
          <span class="flex items-center gap-1">
            <span class="h-1.5 w-1.5 rounded-full bg-status-enrolled"></span>
            Enrolled {{ (probabilities.Enrolled * 100).toFixed(1) }}%
          </span>
          <span class="flex items-center gap-1">
            <span class="h-1.5 w-1.5 rounded-full bg-status-graduate"></span>
            Graduate {{ (probabilities.Graduate * 100).toFixed(1) }}%
          </span>
        </div>
      }
    </div>
  `,
})
export class ConfidenceBarComponent {
  @Input({ required: true }) probabilities!: ClassProbabilities;
  @Input() showLegend = true;

  pct(p: number): number {
    return Math.max(0, Math.min(100, p * 100));
  }
}
