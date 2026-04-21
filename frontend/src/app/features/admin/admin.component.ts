import {
  AfterViewChecked,
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  OnDestroy,
  OnInit,
  NgZone,
  ViewChild,
  computed,
  inject,
  signal,
} from '@angular/core';
import { CommonModule, DatePipe, DecimalPipe } from '@angular/common';
import { FormsModule } from '@angular/forms';

import { SkeletonComponent } from '../../shared/skeleton/skeleton.component';
import { EmptyStateComponent } from '../../shared/empty-state/empty-state.component';
import { MonitoringService } from '../../core/services/monitoring.service';
import { ToastService } from '../../core/services/toast.service';
import {
  ModelRegistryEntry,
  RetrainAuditEntry,
  RetrainResponse,
  RetrainRunStatus,
} from '../../core/models/monitoring.model';

const STAGE_LABELS: Record<string, string> = {
  queued: 'Queued',
  logreg: 'Training logistic regression',
  random_forest: 'Training random forest',
  xgboost: 'Training XGBoost',
  lightgbm: 'Training LightGBM',
  mlp: 'Training MLP',
  evaluate: 'Evaluating & comparing',
  done: 'Complete',
  failed: 'Failed',
};

@Component({
  selector: 'app-admin',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    DatePipe,
    DecimalPipe,
    SkeletonComponent,
    EmptyStateComponent,
  ],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <section class="space-y-6">
      <header>
        <p class="text-2xs uppercase tracking-wider text-brand-300">Administration</p>
        <h1 class="text-2xl lg:text-3xl font-semibold tracking-tight text-white">Retraining &amp; Registry</h1>
        <p class="text-sm text-surface-400 mt-1">
          Trigger a champion-vs-challenger retraining run. Promotion is gated on
          (a) macro-F1 ≥ champion + 1pp, (b) no per-class regression &gt; 2pp, and
          (c) McNemar paired test rejecting the null at α = 0.05.
        </p>
      </header>

      <article class="surface-card p-5 space-y-4">
        <header>
          <h2 class="section-title">Trigger retraining</h2>
          <p class="section-subtitle">Runs in the background — progress and logs stream live below.</p>
        </header>

        <div class="grid grid-cols-1 md:grid-cols-3 gap-3 items-end">
          <div class="md:col-span-2">
            <label class="label" for="reason">Provenance tag</label>
            <input
              id="reason"
              class="input"
              type="text"
              placeholder="e.g. monthly cadence · drift detected · new term cohort"
              [(ngModel)]="reason"
            />
          </div>
          <button type="button" class="btn-primary" [disabled]="busy()" (click)="trigger()">
            @if (busy()) {
              <span class="inline-block h-3 w-3 animate-spin rounded-full border-2 border-white/30 border-t-white"></span>
              Training…
            } @else {
              Retrain now
            }
          </button>
        </div>

        @if (activeRun(); as run) {
          <div class="rounded-md border border-brand-700/40 bg-surface-900/60 p-4 space-y-3">
            <div class="flex items-center justify-between text-xs">
              <div class="flex items-center gap-2">
                @if (run.state === 'running') {
                  <span class="inline-block h-2 w-2 animate-pulse rounded-full bg-brand-400"></span>
                  <span class="font-medium text-brand-200">{{ stageLabel(run.stage) }}</span>
                } @else if (run.state === 'succeeded') {
                  <span class="inline-block h-2 w-2 rounded-full bg-status-graduate"></span>
                  <span class="font-medium text-status-graduate">Run succeeded</span>
                } @else {
                  <span class="inline-block h-2 w-2 rounded-full bg-status-dropout"></span>
                  <span class="font-medium text-status-dropout">Run failed</span>
                }
                <span class="text-2xs text-surface-400 font-mono">run {{ run.run_id }}</span>
              </div>
              <span class="tabular-nums text-surface-300">{{ run.percent }}%</span>
            </div>
            <div class="h-1.5 w-full overflow-hidden rounded-full bg-surface-800">
              <div
                class="h-full transition-all duration-300"
                [style.width.%]="run.percent"
                [class.bg-brand-500]="run.state === 'running'"
                [class.bg-status-graduate]="run.state === 'succeeded'"
                [class.bg-status-dropout]="run.state === 'failed'"
              ></div>
            </div>

            <div
              #logPanel
              class="h-56 overflow-y-auto rounded bg-surface-950/80 border border-surface-800/80 p-3 font-mono text-2xs leading-relaxed text-surface-300"
            >
              @if (logs().length === 0) {
                <p class="text-surface-500">Waiting for trainer output…</p>
              } @else {
                @for (line of logs(); track $index) {
                  <div class="whitespace-pre-wrap" [class.text-status-dropout]="isErrorLine(line)">{{ line }}</div>
                }
              }
            </div>

            @if (run.error) {
              <p class="text-xs text-status-dropout">{{ run.error }}</p>
            }
          </div>
        }

        @if (lastRun(); as r) {
          <div class="rounded-md border border-surface-800/80 bg-surface-900/60 p-4 text-sm">
            <div class="flex items-center justify-between">
              <span class="font-medium" [class.text-status-graduate]="r.promoted" [class.text-status-dropout]="!r.promoted">
                {{ r.promoted ? '✓ Challenger promoted' : '✗ Challenger rejected' }}
              </span>
              <span class="text-2xs text-surface-400">{{ r.timestamp | date: 'medium' }}</span>
            </div>
            <p class="mt-2 text-xs text-surface-300">{{ r.reason }}</p>
            <div class="mt-3 grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
              <div>
                <p class="text-2xs uppercase tracking-wider text-surface-400">Champion F1</p>
                <p class="tabular-nums text-white">{{ (r.champion_macro_f1 * 100) | number: '1.1-1' }}%</p>
              </div>
              <div>
                <p class="text-2xs uppercase tracking-wider text-surface-400">Challenger F1</p>
                <p class="tabular-nums text-white">{{ (r.challenger_macro_f1 * 100) | number: '1.1-1' }}%</p>
              </div>
              <div>
                <p class="text-2xs uppercase tracking-wider text-surface-400">Δ macro-F1</p>
                <p class="tabular-nums" [class]="deltaClass(r.challenger_macro_f1 - r.champion_macro_f1)">
                  {{ ((r.challenger_macro_f1 - r.champion_macro_f1) * 100) | number: '1.2-2' }}pp
                </p>
              </div>
              <div>
                <p class="text-2xs uppercase tracking-wider text-surface-400">McNemar p</p>
                <p class="tabular-nums" [class]="mcnemarClass(r.mcnemar_p_value)">
                  {{ r.mcnemar_p_value == null ? '—' : (r.mcnemar_p_value | number: '1.4-4') }}
                </p>
              </div>
            </div>
          </div>
        }
      </article>

      <article class="surface-card p-5">
        <header class="flex items-center justify-between">
          <div>
            <h2 class="section-title">Retrain audit log</h2>
            <p class="section-subtitle">
              Newest first. Every attempt — promoted or rejected — is preserved.
            </p>
          </div>
          <button type="button" class="btn-secondary" (click)="loadHistory()">↻ Refresh</button>
        </header>

        @if (historyLoading()) {
          <div class="mt-5 space-y-3">
            @for (_ of [1, 2, 3]; track $index) {
              <app-skeleton height="2.25rem" />
            }
          </div>
        } @else if (history().length === 0) {
          <div class="mt-5">
            <app-empty-state icon="◷" title="No retrain history yet"
              description="Trigger a retraining run above and the decision appears here." />
          </div>
        } @else {
          <div class="mt-4 overflow-x-auto">
            <table class="min-w-full text-sm">
              <thead class="bg-surface-850/60 text-2xs uppercase tracking-wider text-surface-400">
                <tr>
                  <th class="text-left py-3 px-4 font-medium">When</th>
                  <th class="text-left py-3 px-4 font-medium">Trigger</th>
                  <th class="text-left py-3 px-4 font-medium">Outcome</th>
                  <th class="text-right py-3 px-4 font-medium">Δ macro-F1</th>
                  <th class="text-right py-3 px-4 font-medium">McNemar p</th>
                  <th class="text-right py-3 px-4 font-medium">n</th>
                  <th class="text-left py-3 px-4 font-medium">Reason</th>
                </tr>
              </thead>
              <tbody class="divide-y divide-surface-800/80">
                @for (e of history(); track e.timestamp + '|' + e.trigger) {
                  <tr class="hover:bg-surface-850/60 align-top">
                    <td class="py-2.5 px-4 text-2xs text-surface-300 whitespace-nowrap">
                      {{ e.timestamp | date: 'short' }}
                    </td>
                    <td class="py-2.5 px-4 text-2xs text-brand-300 font-mono">{{ e.trigger }}</td>
                    <td class="py-2.5 px-4">
                      <span class="badge" [ngClass]="e.promoted ? 'badge-graduate' : 'badge-dropout'">
                        {{ e.promoted ? 'Promoted' : 'Rejected' }}
                      </span>
                    </td>
                    <td class="py-2.5 px-4 text-right tabular-nums" [class]="deltaClass(e.macro_f1_delta)">
                      {{ (e.macro_f1_delta * 100) | number: '1.2-2' }}pp
                    </td>
                    <td class="py-2.5 px-4 text-right tabular-nums" [class]="mcnemarClass(e.mcnemar_p_value)">
                      {{ e.mcnemar_p_value == null ? '—' : (e.mcnemar_p_value | number: '1.4-4') }}
                    </td>
                    <td class="py-2.5 px-4 text-right tabular-nums text-surface-300">{{ e.n_test }}</td>
                    <td class="py-2.5 px-4 text-2xs text-surface-300 max-w-md">{{ e.reason }}</td>
                  </tr>
                }
              </tbody>
            </table>
            @if (promotionRate() != null) {
              <p class="mt-4 text-2xs uppercase tracking-wider text-surface-400">
                Promotion rate (last {{ history().length }}): {{ promotionRate() }}%
              </p>
            }
          </div>
        }
      </article>

      <article class="surface-card p-5">
        <header class="flex items-center justify-between">
          <div>
            <h2 class="section-title">Model registry</h2>
            <p class="section-subtitle">All known versions across stages</p>
          </div>
          <button type="button" class="btn-secondary" (click)="loadRegistry()">↻ Refresh</button>
        </header>

        @if (loading()) {
          <div class="mt-5 space-y-3">
            @for (_ of [1, 2, 3]; track $index) {
              <app-skeleton height="2.25rem" />
            }
          </div>
        } @else if (registry().length === 0) {
          <div class="mt-5">
            <app-empty-state icon="◇" title="Registry is empty" description="Trigger your first retraining run to populate it." />
          </div>
        } @else {
          <div class="mt-4 overflow-x-auto">
            <table class="min-w-full text-sm">
              <thead class="bg-surface-850/60 text-2xs uppercase tracking-wider text-surface-400">
                <tr>
                  <th class="text-left py-3 px-4 font-medium">Name</th>
                  <th class="text-left py-3 px-4 font-medium">Version</th>
                  <th class="text-left py-3 px-4 font-medium">Stage</th>
                  <th class="text-right py-3 px-4 font-medium">Macro-F1</th>
                  <th class="text-right py-3 px-4 font-medium">Dropout recall</th>
                  <th class="text-right py-3 px-4 font-medium">Registered</th>
                </tr>
              </thead>
              <tbody class="divide-y divide-surface-800/80">
                @for (m of registry(); track m.version) {
                  <tr class="hover:bg-surface-850/60">
                    <td class="py-2.5 px-4 text-white">{{ m.name }}</td>
                    <td class="py-2.5 px-4 font-mono text-xs text-brand-300">{{ m.version }}</td>
                    <td class="py-2.5 px-4">
                      <span class="badge" [ngClass]="stageClass(m.stage)">{{ m.stage }}</span>
                    </td>
                    <td class="py-2.5 px-4 text-right tabular-nums text-white">
                      {{ (m.macro_f1 * 100) | number: '1.1-1' }}%
                    </td>
                    <td class="py-2.5 px-4 text-right tabular-nums text-status-graduate">
                      {{ (m.dropout_recall * 100) | number: '1.1-1' }}%
                    </td>
                    <td class="py-2.5 px-4 text-right text-2xs text-surface-400">
                      {{ m.registered_at | date: 'short' }}
                    </td>
                  </tr>
                }
              </tbody>
            </table>
          </div>
        }
      </article>
    </section>
  `,
})
export class AdminComponent implements OnInit, OnDestroy, AfterViewChecked {
  private readonly monitoring = inject(MonitoringService);
  private readonly toast = inject(ToastService);
  private readonly zone = inject(NgZone);

  @ViewChild('logPanel') logPanel?: ElementRef<HTMLDivElement>;
  private stream?: AbortController;
  private shouldAutoscroll = true;

  reason = '';
  readonly busy = signal(false);
  readonly lastRun = signal<RetrainResponse | null>(null);
  readonly loading = signal(true);
  readonly registry = signal<ModelRegistryEntry[]>([]);

  readonly activeRun = signal<RetrainRunStatus | null>(null);
  readonly logs = signal<string[]>([]);

  readonly historyLoading = signal(true);
  readonly history = signal<RetrainAuditEntry[]>([]);
  readonly promotionRate = computed<string | null>(() => {
    const entries = this.history();
    if (!entries.length) return null;
    const promoted = entries.filter((e) => e.promoted).length;
    return ((promoted / entries.length) * 100).toFixed(0);
  });

  ngOnInit(): void {
    this.loadRegistry();
    this.loadHistory();
    // If a run was kicked off from a different tab or before a page refresh
    // we pick it up here and rejoin the log stream rather than showing an
    // empty console while training keeps going on the server.
    this.resumeActiveRun();
  }

  private resumeActiveRun(): void {
    this.monitoring.activeRun().subscribe({
      next: (run) => {
        if (!run) return;
        this.busy.set(true);
        this.logs.set(run.logs ?? []);
        this.activeRun.set(run);
        this.subscribeToRun(run.run_id);
      },
      error: () => {
        // 404/401 here is fine — nothing to resume.
      },
    });
  }

  ngOnDestroy(): void {
    this.stream?.abort();
  }

  ngAfterViewChecked(): void {
    if (this.shouldAutoscroll && this.logPanel) {
      const el = this.logPanel.nativeElement;
      el.scrollTop = el.scrollHeight;
    }
  }

  stageLabel(stage: string): string {
    return STAGE_LABELS[stage] ?? stage;
  }

  isErrorLine(line: string): boolean {
    return /\berror\b|traceback|failed/i.test(line);
  }

  loadRegistry(): void {
    this.loading.set(true);
    this.monitoring.registry().subscribe({
      next: (r) => { this.registry.set(r); this.loading.set(false); },
      error: () => { this.registry.set([]); this.loading.set(false); },
    });
  }

  loadHistory(): void {
    this.historyLoading.set(true);
    this.monitoring.retrainHistory().subscribe({
      next: (h) => { this.history.set(h.entries); this.historyLoading.set(false); },
      error: () => { this.history.set([]); this.historyLoading.set(false); },
    });
  }

  trigger(): void {
    if (this.busy()) return;
    this.busy.set(true);
    this.logs.set([]);
    this.activeRun.set(null);
    this.shouldAutoscroll = true;

    this.monitoring.startRetrain(this.reason || undefined).subscribe({
      next: (start) => {
        this.subscribeToRun(start.run_id);
      },
      error: (err) => {
        this.busy.set(false);
        const detail = err?.error?.detail ?? 'Failed to start retrain';
        this.toast.error('Retrain could not start', detail);
      },
    });
  }

  private subscribeToRun(runId: string): void {
    this.stream?.abort();
    this.stream = this.monitoring.streamRetrainLogs(
      runId,
      (event) => {
        // Event-source callbacks land outside Angular's zone via fetch();
        // hop back so signals trigger change detection normally.
        this.zone.run(() => {
          if (event.snapshot) {
            this.activeRun.set(event.snapshot);
          }
          if (event.event === 'log' && event.line) {
            this.logs.update((lines) => [...lines, event.line!]);
          }
          const run = event.snapshot;
          if (run && (run.state === 'succeeded' || run.state === 'failed')) {
            this.finaliseRun(run);
          }
        });
      },
      () => this.handleStreamDrop(runId),
    );
  }

  /**
   * Reacts to an SSE disconnect that wasn't the planned finalisation path.
   * The server may have completed the job while we were offline, or the
   * proxy may have dropped us mid-flight. Either way, trust the HTTP
   * status endpoint — if the run is still running, silently reconnect; if
   * it's terminal, show the real outcome instead of a scary network toast.
   */
  private handleStreamDrop(runId: string): void {
    this.zone.run(() => {
      this.monitoring.runStatus(runId).subscribe({
        next: (run) => {
          this.activeRun.set(run);
          this.logs.set(run.logs ?? this.logs());
          if (run.state === 'succeeded' || run.state === 'failed') {
            this.finaliseRun(run);
          } else {
            // Still running — reconnect silently.
            this.subscribeToRun(runId);
          }
        },
        error: () => {
          this.busy.set(false);
          this.toast.info('Log stream paused', 'Refresh to resume following this run.');
        },
      });
    });
  }

  private finaliseRun(run: RetrainRunStatus): void {
    this.stream?.abort();
    this.stream = undefined;
    this.busy.set(false);

    if (run.state === 'succeeded' && run.result) {
      this.lastRun.set(run.result);
      this.toast.success(
        run.result.promoted ? 'Challenger promoted' : 'Challenger rejected',
        run.result.reason,
      );
      this.loadRegistry();
      this.loadHistory();
    } else if (run.state === 'failed') {
      this.toast.error('Retrain failed', run.error ?? 'See training logs above');
    }
  }

  stageClass(stage: ModelRegistryEntry['stage']): string {
    switch (stage) {
      case 'Production':
        return 'badge-graduate';
      case 'Staging':
        return 'badge-enrolled';
      default:
        return 'badge-neutral';
    }
  }

  deltaClass(delta: number): string {
    if (delta > 0.001) return 'text-status-graduate';
    if (delta < -0.001) return 'text-status-dropout';
    return 'text-surface-300';
  }

  mcnemarClass(p: number | null): string {
    if (p == null) return 'text-surface-400';
    return p < 0.05 ? 'text-status-graduate' : 'text-status-enrolled';
  }
}
