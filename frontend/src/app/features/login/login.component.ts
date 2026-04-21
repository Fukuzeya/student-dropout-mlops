import { ChangeDetectionStrategy, Component, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule, NgForm } from '@angular/forms';
import { ActivatedRoute, Router, RouterLink } from '@angular/router';

import { AuthService } from '../../core/services/auth.service';
import { ToastService } from '../../core/services/toast.service';
import { environment } from '../../../environments/environment';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterLink],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="min-h-screen flex items-stretch bg-surface-900">
      <!-- Left brand panel -->
      <aside class="hidden lg:flex lg:w-1/2 bg-surface-950 bg-grid-faint bg-grid relative">
        <div class="m-auto px-12 max-w-lg">
          <div class="flex items-center gap-3 mb-10">
            <div class="h-10 w-10 rounded-lg bg-brand-600 flex items-center justify-center text-white font-bold shadow-card">
              UZ
            </div>
            <div>
              <p class="text-base font-semibold text-white">{{ env.appName }}</p>
              <p class="text-xs text-surface-400 uppercase tracking-wider">{{ env.institution }}</p>
            </div>
          </div>
          <h1 class="text-3xl font-semibold tracking-tight text-white">
            Dropout intelligence for the University of Zimbabwe.
          </h1>
          <p class="text-sm text-surface-300 mt-4 leading-relaxed">
            Early-warning scores, SHAP-explained risk factors, and intervention playbooks
            for every student in the cohort.
          </p>
        </div>
      </aside>

      <!-- Right form panel -->
      <main class="flex-1 flex items-center justify-center p-6 lg:p-12">
        <div class="w-full max-w-md space-y-6">
          <header>
            <p class="text-2xs uppercase tracking-wider text-brand-300">Sign in</p>
            <h2 class="text-2xl font-semibold tracking-tight text-white mt-1">Administrator sign-in</h2>
            <p class="text-sm text-surface-400 mt-1">
              Staff credentials are required for retraining and registry promotion.
            </p>
          </header>

          <section class="surface-card p-5 space-y-4">
            <form #f="ngForm" (ngSubmit)="login(f)" class="space-y-3">
              <div>
                <label class="label" for="username">Username</label>
                <input
                  id="username"
                  name="username"
                  class="input"
                  type="text"
                  autocomplete="username"
                  required
                  [(ngModel)]="username"
                />
              </div>
              <div>
                <label class="label" for="password">Password</label>
                <input
                  id="password"
                  name="password"
                  class="input"
                  type="password"
                  autocomplete="current-password"
                  required
                  [(ngModel)]="password"
                />
              </div>
              <button
                type="submit"
                class="btn-primary w-full"
                [disabled]="submitting() || f.invalid"
              >
                @if (submitting()) {
                  <span class="inline-block h-3 w-3 animate-spin rounded-full border-2 border-white/30 border-t-white"></span>
                  Authenticating…
                } @else {
                  Sign in
                }
              </button>
            </form>
          </section>

          <p class="text-2xs text-surface-500 text-center">
            <a routerLink="/dashboard" class="hover:text-brand-300">Continue without signing in →</a>
          </p>
        </div>
      </main>
    </div>
  `,
})
export class LoginComponent {
  protected readonly auth = inject(AuthService);
  private readonly toast = inject(ToastService);
  private readonly router = inject(Router);
  private readonly route = inject(ActivatedRoute);

  readonly env = environment;
  readonly submitting = signal(false);

  username = '';
  password = '';

  login(form: NgForm): void {
    if (form.invalid) {
      return;
    }
    this.submitting.set(true);
    this.auth.login(this.username, this.password).subscribe({
      next: () => {
        this.submitting.set(false);
        this.toast.success('Signed in', `Welcome, ${this.username}.`);
        const redirect = this.route.snapshot.queryParamMap.get('redirect') ?? '/dashboard';
        this.router.navigateByUrl(redirect);
      },
      error: () => {
        this.submitting.set(false);
      },
    });
  }
}
