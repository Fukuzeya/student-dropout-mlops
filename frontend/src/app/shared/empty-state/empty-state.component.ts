import { ChangeDetectionStrategy, Component, Input } from '@angular/core';

@Component({
  selector: 'app-empty-state',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="surface-card p-8 text-center">
      <div class="mx-auto h-12 w-12 rounded-full bg-surface-800 flex items-center justify-center text-xl text-surface-300">
        {{ icon }}
      </div>
      <h3 class="mt-4 text-base font-semibold text-white">{{ title }}</h3>
      @if (description) {
        <p class="mt-1 text-sm text-surface-400 max-w-md mx-auto">{{ description }}</p>
      }
      <ng-content />
    </div>
  `,
})
export class EmptyStateComponent {
  @Input() icon = '∅';
  @Input({ required: true }) title!: string;
  @Input() description?: string;
}
