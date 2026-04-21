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
            Production-grade dropout intelligence for African higher education.
          </h1>
          <p class="text-sm text-surface-300 mt-4 leading-relaxed">
            A bank-grade early-warning system: champion-vs-challenger gating, drift-aware retraining, SHAP-explained
            risk scores, and intervention playbooks tuned for the University of Zimbabwe.
          </p>
          <ul class="mt-8 space-y-2 text-xs text-surface-400">
            <li class="flex items-center gap-2"><span class="text-status-graduate">✓</span> 5-model bake-off · macro-F1 ≥ 0.85</li>
            <li class="flex items-center gap-2"><span class="text-status-graduate">✓</span> DVC reproducible pipeline</li>
            <li class="flex items-center gap-2"><span class="text-status-graduate">✓</span> MLflow registry · Evidently drift</li>
          </ul>
        </div>
      </aside>

      <!-- Right form panel -->
      <main class="flex-1 flex items-center justify-center p-6 lg:p-12">
        <div class="w-full max-w-md space-y-6">
          <header>
            <p class="text-2xs uppercase tracking-wider text-brand-300">Sign in</p>
            <h2 class="text-2xl font-semibold tracking-tight text-white mt-1">Access the EWS console</h2>
            <p class="text-sm text-surface-400 mt-1">
              Read access requires an API key. Administrative actions (retrain, promote) require staff credentials.
            </p>
          </header>

          <!-- API key form -->
          <section class="surface-card p-5 space-y-4">
            <h3 class="section-title text-base">API key</h3>
            <p class="section-subtitle">Used for all read endpoints. Stored locally in your browser.</p>
            <div>
              <label class="label" for="api-key">X-API-Key</label>
              <input
                id="api-key"
                class="input font-mono"
                type="password"
                autocomplete="off"
                placeholder="uz-ews-…"
                [(ngModel)]="apiKey"
              />
            </div>
            <button type="button" class="btn-primary w-full" (click)="saveApiKey()">Save API key</button>
            @if (auth.hasApiKey()) {
              <p class="text-2xs text-status-graduate">✓ API key stored. Read endpoints will authenticate automatically.</p>
            }
          </section>

          <!-- JWT form -->
          <section class="surface-card p-5 space-y-4">
            <h3 class="section-title text-base">Administrator sign-in</h3>
            <p class="section-subtitle">Required for monitoring, retraining, and registry promotion.</p>

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

  apiKey = this.auth.snapshot().apiKey ?? '';
  username = '';
  password = '';

  saveApiKey(): void {
    if (!this.apiKey.trim()) {
      this.toast.warning('API key required', 'Paste your X-API-Key value to continue.');
      return;
    }
    this.auth.setApiKey(this.apiKey);
    this.toast.success('API key saved', 'Read endpoints will authenticate automatically.');
  }

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
