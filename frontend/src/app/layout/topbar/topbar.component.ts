import { ChangeDetectionStrategy, Component, OnInit, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router, RouterLink } from '@angular/router';

import { AuthService } from '../../core/services/auth.service';
import { MonitoringService } from '../../core/services/monitoring.service';
import { ApiHealth } from '../../core/models/monitoring.model';
import { ToastService } from '../../core/services/toast.service';

@Component({
  selector: 'app-topbar',
  standalone: true,
  imports: [CommonModule, RouterLink],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <header class="bg-surface-900/80 backdrop-blur supports-[backdrop-filter]:bg-surface-900/60">
      <div class="px-4 sm:px-6 lg:px-8 h-14 flex items-center gap-4">
        <div class="flex-1 min-w-0">
          <p class="text-xs text-surface-400">Welcome back</p>
          <p class="text-sm font-semibold text-white truncate">
            {{ auth.username() ?? 'Read-only console' }}
            <span class="text-surface-500 font-normal">· Risk Intelligence</span>
          </p>
        </div>

        <div class="hidden lg:flex items-center gap-3">
          <div class="surface-card px-3 py-1.5 flex items-center gap-2">
            <span class="h-1.5 w-1.5 rounded-full" [class.bg-status-graduate]="health()?.status === 'ok'" [class.bg-status-enrolled]="health()?.status === 'degraded'" [class.bg-status-dropout]="health()?.status === 'down'" [class.bg-surface-500]="!health()"></span>
            <span class="text-2xs uppercase tracking-wider text-surface-300">API</span>
            <span class="text-xs font-medium text-white">{{ health()?.status ?? '—' }}</span>
          </div>
          <div class="surface-card px-3 py-1.5">
            <span class="text-2xs uppercase tracking-wider text-surface-400">Model</span>
            <span class="ml-2 text-xs font-mono text-brand-300">{{ health()?.model_version ?? '—' }}</span>
          </div>
          <div class="surface-card px-3 py-1.5">
            <span class="text-2xs uppercase tracking-wider text-surface-400">Macro F1</span>
            <span class="ml-2 text-xs font-semibold tabular-nums text-status-graduate">
              {{ health() ? (health()!.champion_macro_f1 * 100 | number: '1.1-1') + '%' : '—' }}
            </span>
          </div>
        </div>

        <div class="flex items-center gap-2">
          @if (!auth.hasApiKey()) {
            <a routerLink="/login" class="btn-secondary">
              <span class="h-1.5 w-1.5 rounded-full bg-status-enrolled"></span>
              Add API key
            </a>
          }
          @if (auth.isAuthenticated()) {
            <button type="button" class="btn-ghost" (click)="logout()">Sign out</button>
          } @else {
            <a routerLink="/login" class="btn-primary">Admin sign-in</a>
          }
        </div>
      </div>
    </header>
  `,
})
export class TopbarComponent implements OnInit {
  protected readonly auth = inject(AuthService);
  private readonly monitoring = inject(MonitoringService);
  private readonly router = inject(Router);
  private readonly toast = inject(ToastService);

  readonly health = signal<ApiHealth | null>(null);

  ngOnInit(): void {
    this.refresh();
    setInterval(() => this.refresh(), 30_000);
  }

  private refresh(): void {
    this.monitoring.health().subscribe({
      next: (h) => this.health.set(h),
      error: () => this.health.set(null),
    });
  }

  logout(): void {
    this.auth.logout();
    this.toast.info('Signed out', 'Admin session ended.');
    this.router.navigate(['/login']);
  }
}
