import { ChangeDetectionStrategy, Component, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink, RouterLinkActive } from '@angular/router';

import { environment } from '../../../environments/environment';
import { AuthService } from '../../core/services/auth.service';

interface NavLink {
  label: string;
  route: string;
  icon: string;
  description: string;
  adminOnly?: boolean;
}

@Component({
  selector: 'app-sidebar',
  standalone: true,
  imports: [CommonModule, RouterLink, RouterLinkActive],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <aside class="flex h-full flex-col bg-surface-950/60 backdrop-blur">
      <div class="px-5 py-5 border-b border-surface-800/80">
        <div class="flex items-center gap-3">
          <div class="h-9 w-9 rounded-lg bg-brand-600 flex items-center justify-center text-white font-bold shadow-card">
            UZ
          </div>
          <div class="min-w-0">
            <p class="text-sm font-semibold text-white truncate">{{ env.appName }}</p>
            <p class="text-2xs text-surface-400 uppercase tracking-wider">Early Warning</p>
          </div>
        </div>
      </div>

      <nav class="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
        <p class="px-3 pb-2 text-2xs font-semibold uppercase tracking-wider text-surface-500">
          Operate
        </p>
        @for (link of links; track link.route) {
          @if (!link.adminOnly || isAuthed()) {
            <a
              [routerLink]="link.route"
              routerLinkActive="nav-item-active"
              [routerLinkActiveOptions]="{ exact: false }"
              class="nav-item"
            >
              <span
                class="h-7 w-7 rounded-md flex items-center justify-center text-xs bg-surface-800/80 text-surface-300 group-hover:text-white"
              >
                {{ link.icon }}
              </span>
              <span class="flex-1 truncate">{{ link.label }}</span>
            </a>
          }
        }
      </nav>

      <div class="px-3 py-4 border-t border-surface-800/80 space-y-2">
        <div class="surface-card px-3 py-3">
          <p class="text-2xs uppercase tracking-wider text-surface-400">Institution</p>
          <p class="text-xs font-medium text-white mt-1">{{ env.institution }}</p>
          <p class="text-2xs text-surface-400 mt-0.5">Faculty of Science</p>
        </div>
        <p class="text-2xs text-surface-500 px-1">v0.1.0 · Build {{ buildHash }}</p>
      </div>
    </aside>
  `,
})
export class SidebarComponent {
  private readonly auth = inject(AuthService);
  readonly env = environment;
  readonly buildHash = '0001';

  readonly links: NavLink[] = [
    { label: 'Dashboard', route: '/dashboard', icon: '◧', description: 'Risk intelligence overview' },
    { label: 'Students', route: '/students', icon: '◉', description: 'Cohort search & detail' },
    { label: 'Batch Predict', route: '/batch', icon: '⇪', description: 'Score CSV uploads' },
    { label: 'Monitoring', route: '/monitoring', icon: '◔', description: 'Drift, health, KPIs' },
    { label: 'Administration', route: '/admin', icon: '⚙', description: 'Retrain · Promote', adminOnly: true },
  ];

  isAuthed(): boolean {
    return this.auth.isAuthenticated();
  }
}
