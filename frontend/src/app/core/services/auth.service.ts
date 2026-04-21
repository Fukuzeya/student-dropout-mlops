import { Injectable, computed, inject, signal } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable, tap } from 'rxjs';

import { environment } from '../../../environments/environment';
import { AuthState, TokenResponse } from '../models/auth.model';

const STORAGE_KEY = 'uz-ews.auth';

const empty: AuthState = { apiKey: null, jwt: null, username: null, expiresAt: null };

@Injectable({ providedIn: 'root' })
export class AuthService {
  private readonly http = inject(HttpClient);
  private readonly state = signal<AuthState>(this.load());

  readonly snapshot = this.state.asReadonly();
  readonly isAuthenticated = computed(() => {
    const s = this.state();
    if (!s.jwt || !s.expiresAt) {
      return false;
    }
    return Date.now() < s.expiresAt;
  });
  readonly hasApiKey = computed(() => !!this.state().apiKey);
  readonly username = computed(() => this.state().username);

  login(username: string, password: string): Observable<TokenResponse> {
    const body = new HttpParams()
      .set('username', username)
      .set('password', password)
      .set('grant_type', 'password');

    return this.http
      .post<TokenResponse>(`${environment.apiBaseUrl}/auth/token`, body.toString(), {
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      })
      .pipe(
        tap((res) => {
          const expiresAt = Date.now() + res.expires_in * 1000;
          this.update({ jwt: res.access_token, username, expiresAt });
        }),
      );
  }

  setApiKey(apiKey: string): void {
    this.update({ apiKey: apiKey.trim() || null });
  }

  logout(): void {
    this.state.set({ ...empty, apiKey: this.state().apiKey });
    this.persist();
  }

  clearAll(): void {
    this.state.set(empty);
    this.persist();
  }

  private update(partial: Partial<AuthState>): void {
    this.state.update((current) => ({ ...current, ...partial }));
    this.persist();
  }

  private load(): AuthState {
    if (typeof localStorage === 'undefined') {
      return empty;
    }
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) {
        return empty;
      }
      return { ...empty, ...(JSON.parse(raw) as AuthState) };
    } catch {
      return empty;
    }
  }

  private persist(): void {
    if (typeof localStorage === 'undefined') {
      return;
    }
    localStorage.setItem(STORAGE_KEY, JSON.stringify(this.state()));
  }
}
