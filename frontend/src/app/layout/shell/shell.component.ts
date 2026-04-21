import { ChangeDetectionStrategy, Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';

import { SidebarComponent } from '../sidebar/sidebar.component';
import { TopbarComponent } from '../topbar/topbar.component';

@Component({
  selector: 'app-shell',
  standalone: true,
  imports: [RouterOutlet, SidebarComponent, TopbarComponent],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="min-h-screen flex bg-surface-900 bg-grid-faint bg-grid">
      <app-sidebar class="hidden md:flex md:w-60 lg:w-64 shrink-0 border-r border-surface-800/80" />
      <div class="flex-1 flex flex-col min-w-0">
        <app-topbar class="border-b border-surface-800/80" />
        <main class="flex-1 overflow-y-auto bg-panel-gradient">
          <div class="mx-auto max-w-[1400px] px-4 sm:px-6 lg:px-8 py-6 lg:py-8">
            <router-outlet />
          </div>
        </main>
      </div>
    </div>
  `,
})
export class ShellComponent {}
