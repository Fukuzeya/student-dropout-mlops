import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

import { environment } from '../../../environments/environment';
import { AuthService } from './auth.service';
import {
  ApiHealth,
  DashboardKpis,
  DriftAutoRetrainResponse,
  DriftReportSummary,
  DriftRunStart,
  DriftRunStatus,
  EvaluationSummary,
  ModelRegistryEntry,
  RetrainHistoryResponse,
  RetrainResponse,
  RetrainRunStart,
  RetrainRunStatus,
} from '../models/monitoring.model';

export interface RetrainStreamEvent {
  event: string;
  snapshot: RetrainRunStatus;
  line?: string;
}

export interface DriftStreamEvent {
  event: string;
  snapshot: DriftRunStatus;
  line?: string;
}

@Injectable({ providedIn: 'root' })
export class MonitoringService {
  private readonly http = inject(HttpClient);
  private readonly auth = inject(AuthService);
  private readonly base = environment.apiBaseUrl;

  health(): Observable<ApiHealth> {
    return this.http.get<ApiHealth>(`${this.base}/monitoring/health`);
  }

  drift(): Observable<DriftReportSummary> {
    return this.http.get<DriftReportSummary>(`${this.base}/monitoring/drift`);
  }

  kpis(): Observable<DashboardKpis> {
    return this.http.get<DashboardKpis>(`${this.base}/monitoring/kpis`);
  }

  registry(): Observable<ModelRegistryEntry[]> {
    return this.http.get<ModelRegistryEntry[]>(`${this.base}/registry/models`);
  }

  evaluation(): Observable<EvaluationSummary> {
    return this.http.get<EvaluationSummary>(`${this.base}/monitoring/evaluation`);
  }

  triggerRetrain(trigger?: string): Observable<RetrainResponse> {
    const tag = (trigger ?? 'manual-ui').slice(0, 80);
    return this.http.post<RetrainResponse>(
      `${this.base}/retrain`,
      null,
      { params: { trigger: tag } },
    );
  }

  startRetrain(trigger?: string): Observable<RetrainRunStart> {
    const tag = (trigger ?? 'manual-ui').slice(0, 80);
    return this.http.post<RetrainRunStart>(
      `${this.base}/retrain/start`,
      null,
      { params: { trigger: tag } },
    );
  }

  runStatus(runId: string): Observable<RetrainRunStatus> {
    return this.http.get<RetrainRunStatus>(
      `${this.base}/retrain/runs/${runId}`,
    );
  }

  activeRun(): Observable<RetrainRunStatus | null> {
    return this.http.get<RetrainRunStatus | null>(
      `${this.base}/retrain/active`,
    );
  }

  /**
   * Open a server-sent events stream for a retrain run and call `onEvent`
   * for every parsed event. We use fetch() instead of the native EventSource
   * because EventSource cannot send an Authorization header; streaming fetch
   * lets us attach the JWT the same way the rest of the app does.
   *
   * Returns an AbortController the caller uses to tear the stream down
   * (component destroy or run finalisation).
   */
  streamRetrainLogs(
    runId: string,
    onEvent: (event: RetrainStreamEvent) => void,
    onError?: (err: unknown) => void,
  ): AbortController {
    return this._streamSSE<RetrainStreamEvent>(
      `${this.base}/retrain/runs/${runId}/logs`,
      onEvent,
      onError,
    );
  }

  retrainHistory(limit = 50): Observable<RetrainHistoryResponse> {
    return this.http.get<RetrainHistoryResponse>(
      `${this.base}/retrain/history`,
      { params: { limit } },
    );
  }

  driftAutoRetrain(
    file: File,
    threshold: number,
    force: boolean,
  ): Observable<DriftAutoRetrainResponse> {
    const form = new FormData();
    form.append('file', file, file.name);
    return this.http.post<DriftAutoRetrainResponse>(
      `${this.base}/monitoring/drift/auto-retrain`,
      form,
      { params: { threshold: threshold.toFixed(2), force: String(force) } },
    );
  }

  // ------------------------------------------------------------------ drift async
  //
  // Mirrors the retrain async surface. The old synchronous endpoint above
  // kept blocking for the full training window, which looks like a
  // "network error" in the UI when the browser or a proxy cuts the
  // connection. The start/stream/active trio below solves that.

  startDriftAutoRetrain(
    file: File,
    threshold: number,
    force: boolean,
  ): Observable<DriftRunStart> {
    const form = new FormData();
    form.append('file', file, file.name);
    return this.http.post<DriftRunStart>(
      `${this.base}/monitoring/drift/start`,
      form,
      { params: { threshold: threshold.toFixed(2), force: String(force) } },
    );
  }

  driftRunStatus(runId: string): Observable<DriftRunStatus> {
    return this.http.get<DriftRunStatus>(
      `${this.base}/monitoring/drift/runs/${runId}`,
    );
  }

  activeDriftRun(): Observable<DriftRunStatus | null> {
    return this.http.get<DriftRunStatus | null>(
      `${this.base}/monitoring/drift/active`,
    );
  }

  streamDriftLogs(
    runId: string,
    onEvent: (event: DriftStreamEvent) => void,
    onError?: (err: unknown) => void,
  ): AbortController {
    return this._streamSSE<DriftStreamEvent>(
      `${this.base}/monitoring/drift/runs/${runId}/logs`,
      onEvent,
      onError,
    );
  }

  /** Shared SSE reader used by both retrain + drift streams. */
  private _streamSSE<T>(
    url: string,
    onEvent: (event: T) => void,
    onError?: (err: unknown) => void,
  ): AbortController {
    const controller = new AbortController();
    const jwt = this.auth.snapshot().jwt;
    const headers: Record<string, string> = { Accept: 'text/event-stream' };
    if (jwt) headers['Authorization'] = `Bearer ${jwt}`;

    (async () => {
      try {
        const res = await fetch(url, { headers, signal: controller.signal });
        if (!res.ok || !res.body) {
          throw new Error(`SSE connect failed: ${res.status}`);
        }
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          let idx;
          while ((idx = buffer.indexOf('\n\n')) >= 0) {
            const frame = buffer.slice(0, idx);
            buffer = buffer.slice(idx + 2);
            const dataLine = frame.split('\n').find((l) => l.startsWith('data:'));
            if (!dataLine) continue;
            try {
              const payload = JSON.parse(dataLine.slice(5).trim()) as T;
              onEvent(payload);
            } catch {
              // Ignore malformed frames (e.g. keepalive comments).
            }
          }
        }
      } catch (err) {
        if ((err as { name?: string })?.name === 'AbortError') return;
        onError?.(err);
      }
    })();

    return controller;
  }
}
