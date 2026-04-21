import { Injectable, signal } from '@angular/core';

export type ToastVariant = 'success' | 'info' | 'warning' | 'error';

export interface Toast {
  id: number;
  variant: ToastVariant;
  title: string;
  message?: string;
  durationMs: number;
}

@Injectable({ providedIn: 'root' })
export class ToastService {
  private nextId = 1;
  readonly toasts = signal<Toast[]>([]);

  push(variant: ToastVariant, title: string, message?: string, durationMs = 4500): void {
    const toast: Toast = { id: this.nextId++, variant, title, message, durationMs };
    this.toasts.update((current) => [...current, toast]);
    if (durationMs > 0) {
      setTimeout(() => this.dismiss(toast.id), durationMs);
    }
  }

  success(title: string, message?: string): void {
    this.push('success', title, message);
  }

  info(title: string, message?: string): void {
    this.push('info', title, message);
  }

  warning(title: string, message?: string): void {
    this.push('warning', title, message);
  }

  error(title: string, message?: string): void {
    this.push('error', title, message, 7000);
  }

  dismiss(id: number): void {
    this.toasts.update((current) => current.filter((t) => t.id !== id));
  }
}
