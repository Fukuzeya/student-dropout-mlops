import { ChangeDetectionStrategy, Component, computed, inject } from '@angular/core';
import { CommonModule } from '@angular/common';

import { ToastService } from '../../core/services/toast.service';

@Component({
  selector: 'app-toast-container',
  standalone: true,
  imports: [CommonModule],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div
      class="fixed top-4 right-4 z-[9999] flex flex-col gap-2 max-w-sm pointer-events-none"
      aria-live="polite"
      aria-atomic="true"
    >
      @for (t of toasts(); track t.id) {
        <div
          class="pointer-events-auto surface-card-elevated px-4 py-3 animate-fade-in-up flex items-start gap-3"
          [ngClass]="variantClass(t.variant)"
        >
          <span class="mt-0.5 inline-flex h-6 w-6 items-center justify-center rounded-full text-xs font-bold">
            {{ icon(t.variant) }}
          </span>
          <div class="flex-1 min-w-0">
            <p class="text-sm font-semibold text-white truncate">{{ t.title }}</p>
            @if (t.message) {
              <p class="text-xs text-surface-300 mt-0.5 break-words">{{ t.message }}</p>
            }
          </div>
          <button
            type="button"
            class="text-surface-400 hover:text-white transition-colors text-sm leading-none"
            (click)="dismiss(t.id)"
            aria-label="Dismiss notification"
          >×</button>
        </div>
      }
    </div>
  `,
})
export class ToastContainerComponent {
  private readonly service = inject(ToastService);
  readonly toasts = computed(() => this.service.toasts());

  dismiss(id: number): void {
    this.service.dismiss(id);
  }

  variantClass(variant: string): string {
    switch (variant) {
      case 'success':
        return 'border-status-graduate-ring/60';
      case 'warning':
        return 'border-status-enrolled-ring/60';
      case 'error':
        return 'border-status-dropout-ring/60';
      default:
        return 'border-brand-500/40';
    }
  }

  icon(variant: string): string {
    switch (variant) {
      case 'success':
        return '✓';
      case 'warning':
        return '!';
      case 'error':
        return '×';
      default:
        return 'i';
    }
  }
}
