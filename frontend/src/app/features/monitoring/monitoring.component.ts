import {
  AfterViewChecked,
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  NgZone,
  OnDestroy,
  OnInit,
  ViewChild,
  inject,
  signal,
} from '@angular/core';
import { CommonModule, DatePipe, DecimalPipe } from '@angular/common';
import { FormsModule } from '@angular/forms';

import { KpiCardComponent } from '../../shared/kpi-card/kpi-card.component';
import { SkeletonComponent } from '../../shared/skeleton/skeleton.component';
import { EmptyStateComponent } from '../../shared/empty-state/empty-state.component';
import { RigorCardComponent } from './rigor-card.component';

import { MonitoringService } from '../../core/services/monitoring.service';
import { AuthService } from '../../core/services/auth.service';
import { ToastService } from '../../core/services/toast.service';
import {
  ApiHealth,
  DriftReportSummary,
  DriftRunStatus,
  EvaluationSummary,
  ModelRegistryEntry,
} from '../../core/models/monitoring.model';
import { num, pct, timeAgo } from '../../core/utils/format';

// Ordered stage labels for the drift progress bar — mirrors
// backend/app/monitoring/retrain_runs.py:DRIFT_STAGES so the UI text
// tracks whatever the server is reporting.
const DRIFT_STAGE_LABELS: Record<string, string> = {
  queued: 'Queued',
  drift_check: 'Computing drift against reference',
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
  selector: 'app-monitoring',
  standalone: true,
  imports: [
    CommonModule,
    DatePipe,
    DecimalPipe,
    FormsModule,
    KpiCardComponent,
    SkeletonComponent,
    EmptyStateComponent,
    RigorCardComponent,
  ],
  changeDetection: ChangeDetectionStrategy.OnPush,
  templateUrl: './monitoring.component.html',
})
export class MonitoringComponent implements OnInit, OnDestroy, AfterViewChecked {
  private readonly monitoring = inject(MonitoringService);
  private readonly auth = inject(AuthService);
  private readonly toast = inject(ToastService);
  private readonly zone = inject(NgZone);

  @ViewChild('driftLogPanel') driftLogPanel?: ElementRef<HTMLDivElement>;
  private stream?: AbortController;
  private shouldAutoscroll = true;

  readonly health = signal<ApiHealth | null>(null);
  readonly drift = signal<DriftReportSummary | null>(null);
  readonly registry = signal<ModelRegistryEntry[]>([]);
  readonly evaluation = signal<EvaluationSummary | null>(null);
  readonly loading = signal(true);

  readonly autoRetrainFile = signal<File | null>(null);
  readonly autoRetrainThreshold = signal(0.30);
  readonly autoRetrainForce = signal(false);
  readonly autoRetrainBusy = signal(false);

  // Replaces the old synchronous `autoRetrainResult` — the whole snapshot
  // flows through here as stream events arrive, so the UI can show live
  // progress and the final outcome from the same signal.
  readonly driftRun = signal<DriftRunStatus | null>(null);
  readonly driftLogs = signal<string[]>([]);

  readonly isAdmin = this.auth.isAuthenticated;

  fmtPct = pct;
  fmtNum = num;
  fmtAgo = timeAgo;

  ngOnInit(): void {
    this.refresh();
    this.resumeActiveDriftRun();
  }

  ngOnDestroy(): void {
    this.stream?.abort();
  }

  ngAfterViewChecked(): void {
    if (this.shouldAutoscroll && this.driftLogPanel) {
      const el = this.driftLogPanel.nativeElement;
      el.scrollTop = el.scrollHeight;
    }
  }

  refresh(): void {
    this.loading.set(true);
    let pending = 4;
    const done = (): void => {
      pending--;
      if (pending === 0) this.loading.set(false);
    };

    this.monitoring.health().subscribe({
      next: (h) => { this.health.set(h); done(); },
      error: () => { this.health.set(null); done(); },
    });
    this.monitoring.drift().subscribe({
      next: (d) => { this.drift.set(d); done(); },
      error: () => { this.drift.set(null); done(); },
    });
    this.monitoring.registry().subscribe({
      next: (r) => { this.registry.set(r); done(); },
      error: () => { this.registry.set([]); done(); },
    });
    this.monitoring.evaluation().subscribe({
      next: (e) => { this.evaluation.set(e); done(); },
      error: () => { this.evaluation.set(null); done(); },
    });
  }

  manualRefresh(): void {
    this.refresh();
    this.toast.info('Refreshing', 'Pulling latest health, drift, and registry state.');
  }

  driftSeverity(score: number | undefined): { label: string; cls: string } {
    if (score == null) return { label: '—', cls: 'badge-neutral' };
    if (score < 0.1) return { label: 'Stable', cls: 'badge-graduate' };
    if (score < 0.3) return { label: 'Watch', cls: 'badge-enrolled' };
    return { label: 'Drifted', cls: 'badge-dropout' };
  }

  stageClass(stage: ModelRegistryEntry['stage']): string {
    switch (stage) {
      case 'Production':
        return 'badge-graduate';
      case 'Staging':
        return 'badge-enrolled';
      case 'Archived':
        return 'badge-neutral';
      default:
        return 'badge-neutral';
    }
  }

  onAutoRetrainFile(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0] ?? null;
    this.autoRetrainFile.set(file);
  }

  /**
   * Kicks off the drift→auto-retrain pipeline in the background and
   * immediately subscribes to the SSE log stream. Mirrors the retrain
   * page's pattern so the UX is consistent and the request no longer
   * blocks the browser for the full training window.
   */
  runAutoRetrain(): void {
    const file = this.autoRetrainFile();
    if (!file) {
      this.toast.error('No file selected', 'Choose a production batch CSV first.');
      return;
    }
    if (!this.isAdmin()) {
      this.toast.error('Admin login required', 'Sign in as admin to run drift-driven retraining.');
      return;
    }

    this.autoRetrainBusy.set(true);
    this.driftRun.set(null);
    this.driftLogs.set([]);
    this.shouldAutoscroll = true;

    this.monitoring
      .startDriftAutoRetrain(file, this.autoRetrainThreshold(), this.autoRetrainForce())
      .subscribe({
        next: (start) => {
          this.subscribeToDriftRun(start.run_id);
        },
        error: (err) => {
          this.autoRetrainBusy.set(false);
          const detail = err?.error?.detail ?? err?.message ?? 'Auto-retrain could not start.';
          this.toast.error('Auto-retrain failed to start', String(detail));
        },
      });
  }

  private resumeActiveDriftRun(): void {
    this.monitoring.activeDriftRun().subscribe({
      next: (run) => {
        if (!run) return;
        this.autoRetrainBusy.set(true);
        this.driftLogs.set(run.logs ?? []);
        this.driftRun.set(run);
        this.subscribeToDriftRun(run.run_id);
      },
      error: () => {
        // 404/401 here is fine — nothing to resume.
      },
    });
  }

  private subscribeToDriftRun(runId: string): void {
    this.stream?.abort();
    this.stream = this.monitoring.streamDriftLogs(
      runId,
      (event) => {
        // Fetch-based SSE callbacks fire outside Angular's zone; hop back
        // in so signal writes trigger change detection.
        this.zone.run(() => {
          if (event.snapshot) {
            this.driftRun.set(event.snapshot);
          }
          if (event.event === 'log' && event.line) {
            this.driftLogs.update((lines) => [...lines, event.line!]);
          }
          const run = event.snapshot;
          if (run && (run.state === 'succeeded' || run.state === 'failed')) {
            this.finaliseDriftRun(run);
          }
        });
      },
      () => this.handleDriftStreamDrop(runId),
    );
  }

  private handleDriftStreamDrop(runId: string): void {
    this.zone.run(() => {
      this.monitoring.driftRunStatus(runId).subscribe({
        next: (run) => {
          this.driftRun.set(run);
          this.driftLogs.set(run.logs ?? this.driftLogs());
          if (run.state === 'succeeded' || run.state === 'failed') {
            this.finaliseDriftRun(run);
          } else {
            this.subscribeToDriftRun(runId);
          }
        },
        error: () => {
          this.autoRetrainBusy.set(false);
          this.toast.info('Log stream paused', 'Refresh to resume following this run.');
        },
      });
    });
  }

  private finaliseDriftRun(run: DriftRunStatus): void {
    this.stream?.abort();
    this.stream = undefined;
    this.autoRetrainBusy.set(false);

    if (run.state === 'succeeded') {
      if (run.skipped) {
        this.toast.info('No retrain needed', run.skip_reason ?? 'Drift below threshold.');
      } else if (run.retrain?.promoted) {
        this.toast.success('Challenger promoted', 'New champion is live.');
      } else {
        this.toast.info('Retrain ran', run.retrain?.reason ?? 'Champion retained.');
      }
      // Refresh the surrounding cards so drift score, registry, and
      // evaluation numbers reflect the new state.
      this.refresh();
    } else if (run.state === 'failed') {
      this.toast.error('Auto-retrain failed', run.error ?? 'See logs above');
    }
  }

  driftStageLabel(stage: string): string {
    return DRIFT_STAGE_LABELS[stage] ?? stage;
  }

  isErrorLine(line: string): boolean {
    return /\berror\b|traceback|failed/i.test(line);
  }

  deltaClass(delta: number): string {
    if (delta > 0.0001) return 'text-status-graduate';
    if (delta < -0.0001) return 'text-status-dropout';
    return 'text-surface-300';
  }

  mcnemarClass(p: number | null | undefined): string {
    if (p == null) return 'text-surface-300';
    return p < 0.05 ? 'text-status-graduate' : 'text-surface-300';
  }
}
